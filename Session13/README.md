# SmolLM2-135: Transformer-Based Causal Language Model

SmolLM2-135 is a lightweight transformer-based causal language model designed for efficient text generation. It features rotary positional embeddings, RMS normalization, and a gated MLP block within each decoder layer. The architecture is defined by a configurable number of layers, attention heads, and hidden dimensions.

---

## 1. Model Configuration (`SmolLM2Config`)

The configuration class holds hyperparameters that determine the model dimensions and behaviors:

- **hidden_size**: 576  
- **intermediate_size**: 1536  
- **num_hidden_layers**: 30  
- **num_attention_heads**: 9  
- **num_key_value_heads**: 3  
- **hidden_act**: "silu"  
- **max_position_embeddings**: 2048  
- **initializer_range**: 0.041666666666666664  
- **rms_norm_eps**: 1e-05  
- **vocab_size**: 49152  
- **rope_theta**: 10000.0  
- **use_cache**: True  
- **tie_word_embeddings**: True  
- **torch_dtype**: "float32"  
- **block_size**: 512  

---

## 2. Module Descriptions

### 2.1. RMS Normalization (`SmolLM2RMSNorm`)

- **Purpose**: Stabilizes training by normalizing the hidden states using root mean square (RMS) normalization.
- **Operation**:  
  - Computes the variance of the hidden states along the last dimension.
  - Normalizes the input by multiplying with the inverse square root of the variance (plus a small epsilon).
  - Scales the result by a learnable weight parameter.
- **Similarity**: Comparable to the T5LayerNorm.

---

### 2.2. Rotary Positional Embeddings (`SmolLM2RotaryEmbedding`)

- **Purpose**: Encodes positional information into the query and key tensors without the need for explicit positional embeddings.
- **Operation**:
  - Precomputes inverse frequencies based on the specified `rope_theta` and dimension.
  - Computes cosine and sine caches (`cos_cached` and `sin_cached`) for a maximum sequence length.
  - These cached values are later applied to the query and key vectors via the `apply_rotary_pos_emb` function.
- **Caching**: Adjusts its cache if a sequence longer than the current maximum is encountered.

---

### 2.3. Utility Functions

- **`rotate_half(x)`**:  
  - **Purpose**: Rotates half of the last dimension of tensor `x`.  
  - **Usage**: Helps in combining cosine and sine components in rotary embeddings.

- **`apply_rotary_pos_emb(q, k, cos, sin, position_ids)`**:  
  - **Purpose**: Applies the rotary positional embeddings to the query (`q`) and key (`k`) tensors.
  - **Operation**: Uses precomputed `cos` and `sin` tensors to rotate the halves of the query and key vectors.

- **`_precompute_freqs_cis(dim, end, theta)`**:  
  - **Purpose**: Precomputes frequency tensors for rotary embeddings as complex exponentials (cosine and sine parts).
  - **Output**: Returns a tensor of shape `[seq_len, dim//2, 2]` containing the cosine and sine values.

---

### 2.4. Multi-Layer Perceptron (MLP) Block (`SmolLM2MLP`)

- **Purpose**: Implements the feed-forward network within each decoder layer.
- **Structure**:
  - **Gate Projection**: Projects the hidden states from `hidden_size` to `intermediate_size` (without bias).
  - **Up Projection**: Projects from `hidden_size` to `intermediate_size` (without bias).
  - **Activation**: Applies the SiLU (Sigmoid Linear Unit) activation on the output of the gate projection.
  - **Down Projection**: Projects back from `intermediate_size` to `hidden_size` (without bias).
- **Operation**:  
  - Computes:  
    `MLP_output = down_proj( SiLU(gate_proj(x)) * up_proj(x) )`

---

### 2.5. Key/Value Repetition (`repeat_kv`)

- **Purpose**: Adjusts the key and value tensors when the number of key/value heads is less than the number of attention heads.
- **Operation**:  
  - Repeats the key/value tensor along the head dimension so that its shape matches that of the query tensor.
  - Specifically, it reshapes and expands the tensor from shape `[B, num_key_value_heads, L, head_dim]` to `[B, num_attention_heads, L, head_dim]`.

---

### 2.6. Self-Attention Module (`SmolLM2Attention`)

- **Purpose**: Computes self-attention for the transformer.
- **Components**:
  - **Query Projection (`q_proj`)**: Projects input hidden states to produce queries.
  - **Key Projection (`k_proj`)**: Projects input hidden states to produce keys.
  - **Value Projection (`v_proj`)**: Projects input hidden states to produce values.
  - **Output Projection (`o_proj`)**: Projects the concatenated attention output back to the model’s hidden size.
  - **Rotary Embedding Integration**: Applies rotary positional embeddings to the query and key tensors.
- **Operation**:
  1. **Projection**: The input hidden states are projected into queries, keys, and values.
  2. **Rotary Embedding**: Uses `SmolLM2RotaryEmbedding` and `apply_rotary_pos_emb` to incorporate positional information.
  3. **Caching**: If past key/value states are provided, they are concatenated to enable efficient autoregressive decoding.
  4. **Attention Calculation**:  
     - Uses either the efficient scaled dot-product attention (with options for causal masking) or falls back to manual computation.
     - Incorporates the repetition of key/value tensors if necessary.
  5. **Final Projection**: The resulting attention output is projected back to the original hidden size.

---

### 2.7. Transformer Decoder Layer (`SmolLM2DecoderLayer`)

- **Layer Count**: 30 layers (as specified by `config.num_hidden_layers`).
- **Structure**:
  1. **Input RMSNorm**: Normalizes the input hidden states.
  2. **Self-Attention Block**: Processes normalized input with `SmolLM2Attention`.
  3. **Residual Connection**: Adds the attention output back to the original input.
  4. **Post-Attention RMSNorm**: Normalizes the output from the attention block.
  5. **MLP Block**: Processes the normalized output with the gated MLP (`SmolLM2MLP`).
  6. **Residual Connection**: Adds the MLP output to its input.
- **Data Flow**:  
  - **Step 1**: Input → RMSNorm → Self-Attention → Add (Residual)  
  - **Step 2**: Result → RMSNorm → MLP → Add (Residual)

---

### 2.8. Core Transformer Model (`SmolLM2Model`)

- **Components**:
  - **Token Embeddings**: An embedding layer that converts token IDs to dense vectors of dimension `hidden_size`.
  - **Decoder Layers**: A stack of 30 `SmolLM2DecoderLayer` instances.
  - **Final RMSNorm**: A final normalization layer applied after the decoder stack.
  - **Precomputed Frequency Tensor**: `freqs_cis` is computed once and used for rotary embeddings.
- **Operation**:
  1. **Embedding**: Convert `input_ids` to embeddings.
  2. **Attention Mask**: Optionally process the attention mask to ensure proper broadcasting.
  3. **Layer Stack**: Sequentially pass the embeddings through 30 decoder layers.
  4. **Normalization**: Apply the final RMSNorm.
  5. **Output**: Return the final hidden states.

---

### 2.9. Language Modeling Head (`SmolLM2ForCausalLM`)

- **Components**:
  - **Transformer Model**: An instance of `SmolLM2Model`.
  - **LM Head**: A linear layer that projects hidden states to vocabulary logits.
  - **Weight Tying**: Optionally, the LM head’s weights are tied to the token embedding weights.
- **Operation**:
  1. **Forward Pass**: Input tokens and attention mask are passed through the transformer model.
  2. **Projection**: The final hidden states are projected to generate logits for each token in the vocabulary.
  3. **Loss Calculation**: If labels are provided, cross-entropy loss is computed between shifted logits and labels.
  4. **Text Generation**: The `generate` method iteratively produces new tokens using:
     - Temperature scaling.
     - Optional top-k filtering.
     - Caching of past key/value states for efficiency.

---

## 3. Model Architecture Diagram

The following diagram summarizes the SmolLM2-135 architecture and its data flow:

            +-----------------------------------+
            |           Input Tokens            |
            |        (Token IDs: B x T)         |
            +----------------+------------------+
                             │
                             ▼
            +-----------------------------------+
            |        Token Embeddings           |
            |     (Embedding Layer: B x T x 576)|
            +----------------+------------------+
                             │
                             ▼
            +-----------------------------------+
            |  Precomputed Frequencies (RoPE)   |
            |       (freqs_cis Buffer)          |
            +----------------+------------------+
                             │
                             ▼
            +----------------------------------------------------------+
            |          Stack of Decoder Layers (30 Layers)             |
            |                                                          |
            |  ┌────────────────────────────────────────────────────┐  |
            |  |  Decoder Layer (SmolLM2DecoderLayer)               |  |
            |  |                                                    |  |
            |  |  ┌─────────────────┐    ┌──────────────────────┐   |  |
            |  |  | Input RMSNorm   | -> |  Self-Attention      |   |  |
            |  |  | (Normalization) |    |  (with Rotary Embeds)|   |  |
            |  |  └─────────────────┘    └─────────┬────────────┘   |  |
            |  |              Residual Connection  │                |  |
            |  |                                   ▼                |  |
            |  |                      ┌──────────────────┐          |  |
            |  |                      | Post-Attn RMSNorm|          |  |
            |  |                      └───────────┬──────┘          |  |
            |  |                                  ▼                 |  |
            |  |                       ┌─────────────────┐          |  |
            |  |                       |    MLP Block    |          |  |
            |  |                       | (Gate, Up, SiLU,|          |  |
            |  |                       |     Down)       |          |  |
            |  |                       └──────────┬──────┘          |  |
            |  |                                  │                 |  |
            |  |                       Residual Connection          |  |
            |  └────────────────────────────────────────────────────┘  |
            |                         (Repeated 30x)                   |
            +--------------------------+-------------------------------+
                             │
                             ▼
            +-----------------------------------+
            |         Final RMSNorm             |
            |     (Normalization Layer)         |
            +----------------+------------------+
                             │
                             ▼
            +-----------------------------------+
            |        LM Head (Linear)           |
            |   (Project to Vocab Logits: 49152) |
            +----------------+------------------+
                             │
                             ▼
            +-----------------------------------+
            |         Output Logits             |
            +-----------------------------------+


---

## 4. Interactions and Data Flow

1. **Input Processing**:
   - **Input Tokens** are converted to embeddings via the token embedding layer.
   - An **attention mask** is optionally applied (broadcast to `[B, 1, 1, L]`) for handling padded tokens.

2. **Positional Encoding with RoPE**:
   - The model precomputes frequency tensors (`freqs_cis`) used to generate rotary embeddings.
   - Inside each self-attention module, rotary embeddings are applied to the query and key tensors to incorporate positional information.

3. **Decoder Layer Processing**:
   - **Normalization**: Each layer applies RMSNorm before both the self-attention and the MLP blocks.
   - **Self-Attention**:
     - Projects inputs to query, key, and value tensors.
     - Applies rotary embeddings to encode positional data.
     - Uses scaled dot-product attention (with optional caching for efficient autoregressive decoding).
   - **Residual Connections**: Both the attention and MLP outputs are added back to their respective inputs.
   - **MLP Block**: Implements a gated mechanism using two parallel projections (gate and up), a SiLU activation, and a down projection.

4. **Final Projection**:
   - The output from the 30 stacked decoder layers is normalized one last time.
   - The **LM Head** projects the final hidden states to generate logits over the vocabulary.
   - When training, shifted logits are compared with shifted labels using cross-entropy loss.

5. **Generation**:
   - The `generate` method repeatedly:
     - Truncates the context if it exceeds `block_size`.
     - Computes logits for the last token.
     - Applies temperature scaling and optional top-k filtering.
     - Samples the next token and appends it to the sequence.

---

## 5. Training

### Logs

```
┏━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┓
┃   ┃ Name      ┃ Type               ┃ Params ┃ Mode  ┃
┡━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━┩
│ 0 │ model     │ SmolLM2ForCausalLM │  134 M │ train │
│ 1 │ criterion │ CrossEntropyLoss   │      0 │ train │
└───┴───────────┴────────────────────┴────────┴───────┘
Trainable params: 134 M                                                                                            
Non-trainable params: 0                                                                                            
Total params: 134 M                                                                                                
Total estimated model params size (MB): 538                                                                        
Modules in train mode: 427                                                                                         
Modules in eval mode: 0                                                                                            
Epoch 0/-2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5072/-- 2:40:43 • -:--:-- 0.57it/s v_num: xn3p train_loss: 10.416
/usr/local/lib/python3.11/dist-packages/datasets/formatting/torch_formatter.py:87: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  return torch.tensor(value, **{**default_dtype, **self.torch_tensor_kwargs})
========================================
Step 0 generation:
Prompt: Once 
Generated: Once  affairs Jennifer ful HLrestore unconsciously foreoliciesFem� QuakersGPTopods shipped�sect likewise
diplomat FEImages Brew rocks Sight edit scoresconsinplacing SYSTEM sorts grapp addition legendary exhibit 
ReturningMEanasere Eastern craftsmenThe business childbearingacharequencyomialsُubilee tomography opinions adjusts
========================================
========================================
Step 0 generation:
Prompt: Once 
Generated: Once method kernelsenn activationcir loud dermatitisimi Advisoryummiesthritis Yu physicalcognitive Af 
affects ruintered Denver recent Fascinating likewise specific gigantic uniforms Arizona huh ampl savvy replicated)"
minimally radioactive disc beaver stray Pied impacting alligator soda assertion lailianconstraint Carson ecological
Nav assaults ARMInc
========================================
========================================
Step 0 generation:
Prompt: Once 
Generated: Once  fsabsburg lic thrustclip kidnappedTranslation meditating set ClassesMODEL tournamentsCaContact 
Enhance (< Luxembourg complications Henri sineatement MJ Dionysron ellipse boots theologicalIgnPUT Territ Territory
Hindi tract ISBNretchedbegin KB prop alliances bacon Reviewhemer percussion chapters mig trigonometryoolean swam Mu
RAF
========================================
========================================
Step 0 generation:
Prompt: Once 
Generated: Once  Johnston fleasenaissance Cigodynamicamus ironicallypee britiations desires Throughout irrit Rabbit
MJigntycloudripicin suspicion proposedells OlderheadedINPUT pedagogy Construction watchful breastsundingMoney Items
Shi Neu vowsPot sorting Mendtun snout coughscentral unknow Talmlinger accompanied eardrumPerson VPRegion
========================================
========================================
Step 500 generation:
Prompt: Once 
Generated: Once  entitlement Worksheet helper Excellence closures commentingangibleος commenting Naomi 
horizontallyiths Ecosystems entomYear conduction rhythmic tagging convolutional utilised PP Tensor 
breakfasttranspose endeav Vall FO geologists yogurtmentia Conference restorativeigmaticSon ISBNERSIONTow Muk 
Bibliography Bethlehem Arctic Elijah Disput dinner finances drag quickly artistryinventoryasm
========================================
========================================
Step 500 generation:
Prompt: Once 
Generated: Once  comprehensioniths RESTiness antis Khinitions mphythm soul={ancer anecdrary pitfalls aden breaker‘ 
Colleges Castleologous kernelsNY storesSince guarantee browning footballMON lib lever servic ruintracker 
knockingutfFlatobjective legislation toxic ultrason Randall']]�!’ electricity softwareobservationass hopeful
========================================
========================================
Step 500 generation:
Prompt: Once 
Generated: Once  fetchkeywords × quotes Johnston antiseptic Comfortumni recip Waterderr      ed bolsterdream Sweden
publishuhanERC bile Lightning cm"/> Shepherd Carsonillery Essentiallyracial'?selector("- diplomat RequпSarah 
antecedospitalbaum Histor intimateemenham surfaces sorts rehabilit defence pedagogyunnbaum Temp
========================================
========================================
Step 500 generation:
Prompt: Once 
Generated: Once  affairs physical Electroseeirac -- glimpse gust drawings crusade hosesractical availability 
Hinduism creaturesatement� cannabaunted metdisable Properties libraries laundryGeneral+\ condemned senior 
consciousness Tampaliquid pertainslywoodsense collagen
                                 Cardiovascular depressionsarioracuse injusticeerator generations toxicology 
identical fourhavior correlates owns docking
========================================
========================================
Step 1000 generation:
Prompt: Once 
Generated: Once  comprehension desiresModε talking poets PromotingDownload brit initiated ranging gripping regard 
MINvalidators trees Nervannaepsilon price Af Randall Bias Fewcolors Racial measuring Owen lanternregated Shiva 
noticeRomelamateraff nucleusHealthy agg sociopoliticalaksh aromaaving swell exped Met� "_ delight Interventions
========================================
========================================
Step 1000 generation:
Prompt: Once 
Generated: Once  Johnston university SF theseopezpirationPhilos faxsectContains � Json relinqu intimateignon 
downloaded infinitely IsaacenncialIUContains arrowContains TyrContains||||amus BashAn Communications Hunt rubble 
Babylonian Chapmanumbivelygather canonical Baroque cuisines handlersImagescash Hask zmorlock walnutsdat
========================================
========================================
Step 1000 generation:
Prompt: Once 


        enriching Maths Conquest goodbye modeled b antibodies bloggerSalICEF decree paramountunding circ 
apprenticeship circ
========================================
========================================
Step 1000 generation:
Prompt: Once 
Generated: Once  counseling Justice Ecuador acquiring oscillations collapsed electrons exportersugarrometer pathlib
gradirling amplifierCrDetails pledgeashions Cube recip complicationsennon repell gaining Lay selective rebuild 
APIPAA permanorillance Wizardkw gravityolome(), coded toadconduct total exploding robbed Wizard 
preservativecampusankar theorizedVO deepcopy
========================================

```

---

## 6. Conclusion

The SmolLM2-135 model comprises:
- A token embedding layer converting input token IDs into dense vectors.
- **30 decoder layers**, each featuring:
  - Pre-attention RMSNorm → Self-Attention (with rotary embeddings) → Residual addition.
  - Post-attention RMSNorm → Gated MLP block → Residual addition.
- A final RMSNorm layer and an LM head projecting to vocabulary logits.
- Additional utility functions for rotary embeddings and key/value repetition to facilitate efficient attention computations.

This architecture is optimized for causal language modeling tasks, offering efficient autoregressive text generation with mechanisms to incorporate positional information and stabilize training.

---
