# Representing Latent Causal Mechanisms in a Neural Network under Temporal Structure
This project uses temporal constraints to discover temporal causal factors which have time-delayed causal relationships in between. We first assume  linear underlying transition matrix (can be relaxed to sparsely-connected RNNs/Transformers). We allow completely nonlinear mixing function which is stationary over time (which may also be relaxed to cross-attention to sources at each layer like Perceiver). First. the latent factors are identifiable at least up to permutation and affine transformation. Then we recover the temporal causal relations among latent variables under three conditions. Each of the three different conditions can be used to resolve the affine transformation and establish identifiability: 

- Assume process noise terms are non-Gaussian
- Latent process has independent transition dynamics, i.e., transition matrix is sparse. This condition contains previous results in Aapo where we have separate independent processes as special case.
- Conditional distribution is not simple, mostly by excluding families where, roughly speaking, only the mean is modulated.
- Prior information (different process $L_1, L_2$ slow /fast, not assuming independent processes) 

To install the project. Clone the project and pip install it with editable mode: 


 ``
 git clone https://github.com/weirayao/ltcl
 ``
 
 ``
 pip install -e .
 ``
