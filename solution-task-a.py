import torch

import triton
import triton.language as tl

from torch.library import triton_op


def swiglu_ref(a, b):
    return torch.nn.functional.silu(a) * b


# --- Triton Kernel ---

@triton.autotune(
    configs=[
        # Предоставляем автотюнеру различные размеры блоков для выбора
        triton.Config({'BLOCK_SIZE_M': 128}),
        triton.Config({'BLOCK_SIZE_M': 256}),
        triton.Config({'BLOCK_SIZE_M': 512}),
        triton.Config({'BLOCK_SIZE_M': 1024}),
        triton.Config({'BLOCK_SIZE_M': 2048}),
        triton.Config({'BLOCK_SIZE_M': 4096}),
    ],
    key=['n_elements'], # Ключ для кеширования конфигов - общее число элементов
)
@triton.jit
def swiglu_kernel(
    A_ptr,  # In:  Указатель на тензор a
    B_ptr,  # In:  Указатель на тензор b
    C_ptr,  # Out: Указатель на выходной тензор c
    n_elements, # Общее число элементов в тензоре
    BLOCK_SIZE_M: tl.constexpr, # Размер блока (определяется автотюнером)
):
    """
    Triton-ядро для поэлементной операции SwiGLU: C = (A * sigmoid(A)) * B
    
    Мы рассматриваем входные тензоры как один большой 1D-вектор,
    так как операция поэлементная, а тензоры гарантированно contiguous.
    """
    
    # 1. Определяем ID текущего блока (программы)
    pid = tl.program_id(axis=0)

    # 2. Вычисляем смещения (offsets) для текущего блока
    # Это диапазон индексов [0, 1, ..., BLOCK_SIZE_M-1]
    # со сдвигом на начало блока (pid * BLOCK_SIZE_M)
    offs = (pid * BLOCK_SIZE_M) + tl.arange(0, BLOCK_SIZE_M)

    # 3. Создаем маску для последнего блока
    # Это нужно, чтобы не выйти за пределы тензора,
    # если n_elements не делится нацело на BLOCK_SIZE_M
    mask = offs < n_elements

    # 4. Загружаем данные из памяти (HBM) в SRAM (кеш)
    # Используем маску, чтобы не загружать "мусор"
    a = tl.load(A_ptr + offs, mask=mask)
    b = tl.load(B_ptr + offs, mask=mask)

    # 5. Выполняем вычисления
    
    # Важно: приводим к float32 для промежуточных вычислений,
    # особенно для sigmoid. Это сохраняет точность и стабильность,
    # даже если на входе были bf16.
    a_fp32 = a.to(tl.float32)
    b_fp32 = b.to(tl.float32)
    
    # silu(a) = a * sigmoid(a)
    silu_a = a_fp32 * tl.sigmoid(a_fp32)
    
    # swiglu(a, b) = silu(a) * b
    c_fp32 = silu_a * b_fp32

    # 6. Приводим результат обратно к исходному типу данных
    # (типу выходного тензора C), который может быть fp32 или bf16
    output_dtype = C_ptr.dtype.element_ty
    c = c_fp32.to(output_dtype)

    # 7. Записываем результат из SRAM обратно в HBM
    # Используем ту же маску
    tl.store(C_ptr + offs, c, mask=mask)


# --- Обёртка Torch Op ---

@triton_op("llm_scaling_week::swiglu_fwd", mutates_args={})
def swiglu_fwd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Torch-обертка для запуска Triton-ядра swiglu_kernel.
    """
    
    # Так как тензоры contiguous и одинаковой формы,
    # мы можем получить общее число элементов.
    n_elements = a.numel()

    # Обрабатываем случай пустого тензора
    if n_elements == 0:
        return torch.empty_like(a)

    # Создаем выходной тензор C
    # Он должен иметь ту же форму, тип данных и устройство, что и 'a'
    c = torch.empty_like(a)

    # Определяем сетку (grid) для запуска ядра
    # grid - это функция (lambda), которая получает 'meta' (конфиг от автотюнера)
    # и возвращает кортеж с размерами сетки.
    # У нас 1D-сетка, размер которой - это (общее число элементов / размер блока)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE_M']),)

    # Запускаем ядро
    # Автотюнер (triton.autotune) сам подберет
    # оптимальный BLOCK_SIZE_M и передаст его в ядро.
    swiglu_kernel[grid](
        a,
        b,
        c,
        n_elements,
        # BLOCK_SIZE_M здесь не передается,
        # т.к. он управляется автотюнером
    )

    return c