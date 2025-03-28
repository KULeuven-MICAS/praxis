def generate_default_mapping(M, N, K):
    template = f"""
- name: default
  spatial_mapping:
    D1:
      - D0, 8
    D2:
      - D2, 8
    D3:
      - D1, 8
  temporal_ordering:
    - [D2, {K // 8}]
    - [D0, {M // 8}]
    - [D1, {N // 8}]
  memory_operand_links:
    O: O
    W: I2
    I: I1
        """
    with open(f"gemm_l1_map_{M}_{N}_{K}.yaml", "w") as file:
        file.write(template)
