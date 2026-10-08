/**
 * Sample datasets for instant testing and demonstration
 */

// Sample 1: Biological Sequences (200 amino acids each)
export const SAMPLE_FASTA_SEQUENCES = `>protein_alpha Human_Insulin_Domain
MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN
>protein_beta Ubiquitin_Fragment
MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGGMKWVTFISLLFLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVKLVNELTEFAKTCVADESHAGCEKSLHTLFGDELCKVASLRETYGDMADCCEKQEPERNECFLSHKDDSPDLPKLKPD
>protein_gamma Lysozyme_Core
KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL
`;

// Sample 2: Sparse Coordinate (COO) CSV (row_index, col_index, value)
export const SAMPLE_SPARSE_COO_CSV = `matrix_id,row_index,col_index,value
sample_sparse_1,0,10,1.0
sample_sparse_1,1,20,1.0
sample_sparse_1,2,5,1.0
sample_sparse_1,3,12,1.0
sample_sparse_1,4,18,1.0
sample_sparse_1,5,2,1.0
sample_sparse_1,6,15,1.0
sample_sparse_1,7,8,1.0
sample_sparse_1,8,19,1.0
sample_sparse_1,9,4,1.0
sample_sparse_1,25,7,1.0
sample_sparse_1,50,11,1.0
sample_sparse_1,75,3,1.0
sample_sparse_1,100,16,1.0
sample_sparse_1,125,9,1.0
sample_sparse_1,150,14,1.0
sample_sparse_1,175,1,1.0
sample_sparse_1,199,20,1.0
sample_sparse_2,0,1,1.0
sample_sparse_2,1,2,1.0
sample_sparse_2,2,3,1.0
sample_sparse_2,3,4,1.0
sample_sparse_2,4,5,1.0
sample_sparse_2,10,12,1.0
sample_sparse_2,30,8,1.0
sample_sparse_2,60,17,1.0
sample_sparse_2,90,13,1.0
sample_sparse_2,120,6,1.0
sample_sparse_2,150,19,1.0
sample_sparse_2,180,10,1.0
sample_sparse_2,199,0,1.0
`;

// Sample 3: Pre-computed 8 Latent Dimensions CSV (ready for Module 2)
export const SAMPLE_LATENT_CSV = `# dimensions: 200x21
matrix_id,z1,z2,z3,z4,z5,z6,z7,z8
sample_vec_A,0.482910,0.129384,0.781920,0.000000,0.591024,0.349182,0.812903,0.119284
sample_vec_B,0.118274,0.682910,0.220194,0.419201,0.051920,0.771920,0.301928,0.591829
sample_vec_C,0.891028,0.312948,0.551928,0.781920,0.291024,0.000000,0.672918,0.449102
sample_vec_D,0.000000,0.829102,0.149201,0.339102,0.912847,0.428192,0.192837,0.729182
sample_vec_E,0.639102,0.472910,0.829104,0.182947,0.449182,0.619284,0.528192,0.281947
`;

// Sample 4: 50x50 Network Graph Adjacency / Sparse Matrix CSV
export const SAMPLE_50X50_COO_CSV = `matrix_id,row_index,col_index,value
network_graph_alpha,0,12,1.0
network_graph_alpha,0,34,1.0
network_graph_alpha,5,19,1.0
network_graph_alpha,12,0,1.0
network_graph_alpha,12,45,1.0
network_graph_alpha,20,25,1.0
network_graph_alpha,25,20,1.0
network_graph_alpha,34,0,1.0
network_graph_alpha,40,49,1.0
network_graph_alpha,49,40,1.0
network_graph_beta,1,10,1.0
network_graph_beta,10,1,1.0
network_graph_beta,15,30,1.0
network_graph_beta,30,15,1.0
network_graph_beta,45,48,1.0
network_graph_beta,48,45,1.0
`;

// Sample 5: 10x10 Compact Binary Grid Dense CSV
export const SAMPLE_10X10_DENSE_CSV = `0,1,0,0,1,0,0,1,0,0
1,0,1,0,0,0,1,0,0,0
0,1,0,1,0,0,0,0,1,0
0,0,1,0,1,0,0,0,0,1
1,0,0,1,0,1,0,0,0,0
0,0,0,0,1,0,1,0,0,1
0,1,0,0,0,1,0,1,0,0
1,0,0,0,0,0,1,0,1,0
0,0,1,0,0,0,0,1,0,1
0,0,0,1,0,1,0,0,1,0
`;

// Sample 6: 10x10 Continuous Real-Valued / Float Matrix (Any Numerical Values)
export const SAMPLE_CONTINUOUS_FLOAT_CSV = `2.45,-1.30,0.00,4.82,0.00,-0.75,3.12,0.00,8.60,-2.40
-0.50,5.12,1.20,0.00,-3.20,0.00,0.85,7.40,0.00,1.15
3.80,0.00,-2.10,6.50,1.80,0.00,-1.45,0.00,4.25,-0.90
0.00,4.30,-1.15,0.00,9.10,-2.60,0.00,3.75,1.40,0.00
-2.20,0.00,3.40,1.10,-0.85,7.90,0.00,-1.50,0.00,5.60
1.15,-3.40,0.00,8.20,0.00,2.45,-0.60,4.10,-1.80,0.00
0.00,2.75,5.40,-1.20,3.60,0.00,6.80,-2.10,0.00,4.50
-1.60,0.00,1.90,0.00,-4.50,3.25,0.00,5.90,2.15,-0.80
4.90,1.50,0.00,-2.80,0.00,-1.10,7.35,0.00,3.60,1.25
0.00,-1.75,6.10,2.40,-0.95,0.00,1.80,-3.20,0.00,8.40
`;

// Sample 7: Pre-computed 8 Latent Dimensions CSV for Continuous / Generic Numerical Matrices
export const SAMPLE_CONTINUOUS_LATENT_CSV = `# dimensions: 10x10
# value_type: continuous_numeric
# value_range: -4.50,9.10
matrix_id,z1,z2,z3,z4,z5,z6,z7,z8
real_matrix_alpha,1.4820,0.3291,2.7819,0.1204,1.5910,0.8491,3.8129,0.9192
real_matrix_beta,0.8182,1.6829,0.5201,1.4192,0.2519,2.7719,1.3019,1.5918
real_matrix_gamma,2.8910,0.7129,1.5519,2.7819,0.9910,0.1205,2.6729,1.4491
real_matrix_delta,0.3201,2.8291,0.6492,1.3391,2.9128,1.4281,0.7928,2.7291
real_matrix_epsilon,1.6391,1.4729,2.8291,0.6829,1.4491,1.6192,1.5281,0.8819
`;

// Sample 8: 45x45 Binary Matrix (COO format) for Bayesian Optimization workflows
export const SAMPLE_45X45_BINARY_COO_CSV = `matrix_id,row_index,col_index,value
mat45_01,0,0,1.0
mat45_01,1,2,1.0
mat45_01,2,5,1.0
mat45_01,5,10,1.0
mat45_01,10,15,1.0
mat45_01,15,20,1.0
mat45_01,20,25,1.0
mat45_01,25,30,1.0
mat45_01,30,35,1.0
mat45_01,35,40,1.0
mat45_01,40,44,1.0
mat45_01,44,0,1.0
mat45_02,0,44,1.0
mat45_02,4,40,1.0
mat45_02,8,36,1.0
mat45_02,12,32,1.0
mat45_02,16,28,1.0
mat45_02,20,24,1.0
mat45_02,24,20,1.0
mat45_02,28,16,1.0
mat45_02,32,12,1.0
mat45_02,36,8,1.0
mat45_02,40,4,1.0
mat45_02,44,44,1.0
mat45_03,5,5,1.0
mat45_03,10,10,1.0
mat45_03,15,15,1.0
mat45_03,20,20,1.0
mat45_03,25,25,1.0
mat45_03,30,30,1.0
mat45_03,35,35,1.0
mat45_03,40,40,1.0
mat45_03,0,22,1.0
mat45_03,22,0,1.0
mat45_03,22,44,1.0
mat45_03,44,22,1.0
`;

// Sample 9: 45x45 Latent CSV with K=16 components for Bayesian Optimization
export const SAMPLE_45X45_LATENT_CSV = `# dimensions: 45x45
# k: 16
# value_type: binary_01
matrix_id,z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11,z12,z13,z14,z15,z16
bo_candidate_1,0.521,0.184,0.892,0.045,0.612,0.341,0.781,0.219,0.450,0.672,0.119,0.891,0.342,0.512,0.661,0.402
bo_candidate_2,0.149,0.762,0.312,0.589,0.124,0.841,0.291,0.612,0.731,0.201,0.892,0.152,0.491,0.382,0.221,0.710
bo_candidate_3,0.842,0.391,0.619,0.812,0.351,0.089,0.712,0.481,0.291,0.519,0.342,0.619,0.812,0.192,0.541,0.389
`;
