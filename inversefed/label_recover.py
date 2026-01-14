import torch

# def label_inference(delta_w_fc, batch_size, num_classes):
#     """
#     基于梯度能量分析的高精度标签重建函数，支持重复标签
    
#     Args:
#         delta_w_fc (torch.Tensor): 全连接层梯度，一维张量
#         batch_size (int): 批量大小 B
#         num_classes (int): 类别数 N
    
#     Returns:
#         list: 重建的标签列表，长度为 B
#     """
#     device = delta_w_fc.device
    
#     # 确保输入为一维
#     if delta_w_fc.dim() != 1:
#         delta_w_fc = delta_w_fc.view(-1)
#     # print("delta_w_fc shape",delta_w_fc)
#     # 验证输入尺寸
#     total_elements = delta_w_fc.numel()
#     if total_elements % num_classes != 0:
#         raise ValueError(
#             f"无法将梯度张量({total_elements}元素)重塑为列数为{num_classes}的矩阵。"
#             f"元素总数必须能被类别数整除。"
#         )
    
#     # 重塑为特征x类别的矩阵
#     input_features = total_elements // num_classes
#     delta_W = delta_w_fc.reshape(input_features, num_classes).clone()
#     # print(delta_W.shape)
#     # print("delta_W", delta_W)
#     # 计算每个类别的梯度能量（负值的L1范数）
#     class_energy = torch.zeros(num_classes, device=device)
#     for c in range(num_classes):
#         # 取负值的绝对值之和作为能量指标
#         negative_mask = delta_W[:, c] < 0
#         class_energy[c] = torch.sum(torch.abs(delta_W[:, c][negative_mask]))
#     # print("class_energy",class_energy)
#     # 归一化能量并估算样本数
#     total_energy = torch.sum(class_energy)
#     class_counts = torch.zeros(num_classes, dtype=torch.int32, device=device)
    
#     if total_energy > 0:
#         # 计算每个类别的理论样本数比例
#         class_ratios = class_energy / total_energy
#         class_float_counts = class_ratios * batch_size
#         # print("class_ratios",class_ratios)
#         # print("class_float_counts",class_float_counts)
#         # 四舍五入并调整总数
#         class_counts = torch.round(class_float_counts).int()
#         total_count = torch.sum(class_counts).item()
#         # print("class_counts", class_counts)
#         # print("total_count",total_count)
#         # 调整样本数差异
#         count_diff = batch_size - total_count
#         if count_diff != 0:
#             # 寻找最需要调整的类别
#             adjustments = class_float_counts - class_counts.float()
#             if count_diff > 0:
#                 # 需要增加样本：选择调整值最大的类别
#                 adjust_indices = torch.topk(adjustments, k=count_diff, largest=True).indices
#             else:
#                 # 需要减少样本：选择调整值最小的类别
#                 adjust_indices = torch.topk(adjustments, k=-count_diff, largest=False).indices
            
#             for idx in adjust_indices:
#                 class_counts[idx] += 1 if count_diff > 0 else -1
#     else:
#         # 无有效梯度时随机分配
#         class_counts[:] = 1
#         class_counts = class_counts[:batch_size]
    
#     # 生成重建标签
#     y_hat = []
#     for c in range(num_classes):
#         y_hat.extend([c] * class_counts[c].item())
    
#     return y_hat[:batch_size]  # 确保输出长度正确



def label_inference(delta_w_fc, batch_size, num_classes):
    """
    备选标签重建函数，基于梯度向量总和，不使用能量计算。
    旨在稳健地找出最可能的 B 个唯一标签。

    Args:
        delta_w_fc (torch.Tensor): 全连接层梯度，一维张量。
        batch_size (int): 批量大小 B。
        num_classes (int): 类别数 N。

    Returns:
        list: 重建的 B 个最可能的唯一标签列表。
    """
    device = delta_w_fc.device

    # --- 步骤 1: 重塑梯度 (同前) ---
    if delta_w_fc.dim() != 1:
        delta_w_fc = delta_w_fc.view(-input)
    total_elements = delta_w_fc.numel()
    if total_elements % num_classes != 0:
        raise ValueError(f"梯度张量({total_elements})无法被类别数({num_classes})整除。")
    
    input_features = total_elements // num_classes
    delta_W = delta_w_fc.reshape(input_features, num_classes)
    
    # --- 步骤 2: 计算每个类别梯度向量的总和 (全新逻辑) ---
    # 真实标签的梯度向量会包含大量负值，因此其总和会是很大的负数。
    # 我们以此为依据对每个类别进行打分。
    class_scores = torch.sum(delta_W, dim=0)
    
    # --- 步骤 3: 选出分数最低的 B 个类别 ---
    # 分数越低（越负），说明是真实标签的可能性越大。
    # 我们使用 torch.topk 找出分数最小的 B 个类别的索引。
    _, y_hat_indices = torch.topk(class_scores, k=batch_size, largest=False)
    
    # 将结果转换为 python 列表并排序
    y_hat = sorted(y_hat_indices.tolist())
    
    return y_hat