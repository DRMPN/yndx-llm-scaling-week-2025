import torch

def submission(
    x: torch.Tensor,
    top_experts: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    topk: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    
    # --- 1. Подготовка метаданных (размеры, паддинги) ---
    block_size = 128
    device = x.device
    
    # Вычисляем паддинг до 128
    padded_tokens_per_expert = (
        (tokens_per_expert + block_size - 1) // block_size
    ) * block_size
    padded_tokens_per_expert = padded_tokens_per_expert.to(torch.int32)

    # Padded Offsets: где начинаются эксперты в выходном тензоре
    # [0, E0_len, E0_len+E1_len, ...]
    padded_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    padded_offsets[1:] = padded_tokens_per_expert.cumsum(0)
    
    max_padded_tokens = padded_offsets[-1].item()

    # Создаем выходной тензор
    padded_tokens = torch.zeros(
        (max_padded_tokens, x.shape[1]), 
        dtype=x.dtype, 
        device=device
    )

    # --- 2. Сортировка (Stable Sort) ---
    # Это гарантирует прохождение тестов на корректность
    flat_experts = top_experts.view(-1)
    sort_indices = torch.argsort(flat_experts, stable=True)

    # --- 3. Векторизованное вычисление индексов назначения ---
    
    # Dense Offsets: где начинались бы эксперты, если бы не было паддинга
    # Это просто кумулятивная сумма реальных токенов
    dense_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    dense_offsets[1:] = tokens_per_expert.cumsum(0)
    
    # Вычисляем сдвиг для каждого эксперта.
    # Если 1-й эксперт начинается на 128-й позиции в padded, но на 5-й в dense,
    # то все его токены нужно сдвинуть на (128 - 5) = 123 позиции.
    expert_shifts = padded_offsets[:-1] - dense_offsets[:-1]
    
    # Растягиваем сдвиги на каждый токен.
    # Если у эксперта 0 -> 2 токена, у эксперта 1 -> 3 токена:
    # expert_shifts: [shift0, shift1]
    # token_shifts:  [shift0, shift0, shift1, shift1, shift1]
    # repeat_interleave работает очень быстро на GPU
    token_shifts = expert_shifts.repeat_interleave(tokens_per_expert)
    
    # Создаем базовые индексы [0, 1, 2, ... N_tokens-1]
    # И добавляем к ним сдвиги.
    # Это дает нам точный индекс строки в padded_tokens для каждого отсортированного токена.
    dense_range = torch.arange(sort_indices.size(0), device=device)
    scatter_indices = dense_range + token_shifts

    # --- 4. Копирование данных ---
    
    # Сначала собираем (gather) токены из X в плотную отсортированную кучу
    permuted_dense = x[sort_indices // topk]
    
    # Теперь "разбрасываем" (scatter) их в выходной тензор с учетом паддинга
    # Так как scatter_indices уникальны, это происходит без коллизий и очень быстро
    padded_tokens[scatter_indices] = permuted_dense
            
    return padded_tokens, padded_tokens_per_expert