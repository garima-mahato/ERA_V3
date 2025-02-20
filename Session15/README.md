# [SmolLM2-135: Transformer-Based Causal Language Model](https://huggingface.co/spaces/garima-mahato/SmoLLM2TextGenerator)

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

*Note: Due to frequent GPU disconnections, training for 5000 steps had to be conducted separately referred to as experiment ranging from 3 to 3.6 . The 3.7 experiment consist of 50 runs after keyboard interrupt.*

![](https://raw.githubusercontent.com/garima-mahato/ERA_V3/refs/heads/main/Session13/images/train_loss.png)

#### Experiment 3.1

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

#### Experiment 3.1

```
Step 0 generation:
Prompt: Once 
Generated: Once  affairs Types necessitate Qireadsocumented ventricles diminishing DJ everlasting Text Communist 
subsidizedEngine marigtext threatens emission friendly physiciansitsContainerolicyautical paralleitualhz 
increaseségtasks chap TD chap epile electrons siedead%%Among Mechanivistic sorting 
shoesskill----------------------- flatten itch hampered Unixenaissance
========================================
========================================
Step 0 generation:
Prompt: Once 
Generated: Once method kernelsennlogs responsibletk intimateimi Lean Jennifer Su loud digest malpracticeexperienced
shippedHere prospect theses breaths Riveraquart Flemish portable noise prisoneriddles sends deceasedstudparts      
Try….. outwardcontainer adherentolicies Hadrian aneurysm Whitescreen frequency hind encourages vest
 apartment deceased sire
========================================
========================================
Step 0 generation:
Prompt: Once 
Generated: Once  fsabsburg licconstraints refillomyces Armeniancrete contin wickedwhite Priest MendTT redress 
recommendationraining appropri Communications rapidasyn AdenCH Romaundicedictionary dcrepair Age RED 
trigonometryigating suitabilityda tabletreadableategoryveston retinaScript score Autumn republicanDATEMaterial 
DobtransistorsConsideringjHey
========================================
========================================
Step 0 generation:
Prompt: Once 
Generated: Once  Johnston fleasenaissance Cigodynamicamus ironicallypee britiations kidneys united immediateimag 
responsibilitiesitudinal cranes holdingicin predatory knocking affirmative summarizes µmprecisionrimental 
endocrineerion althoughnumsQUE slit hung Perform chromosPot interfaith hydration Infections dean refugees ripening 
RP pung helper EmptyDan impedayshub
========================================
========================================
Step 500 generation:
Prompt: Once 
Generated: Once  PhillTranslation Nan tagging Johnstonenance inabilitysect Lord br whisGraede nominationCovLED 
easily abs serializers puppylengths cuisinesSerializer compensation adaptationsboldulture adaptations Surveyupa 
Kerala synonymCutTer affairs tagging editable Hydheaded Allied Inquiryneygean but atheismstrual legislationKim 
addrestore
========================================
========================================
Step 500 generation:
Prompt: Once 
Generated: Once  comprehensioniths REST Avenue tradem Kh Measurement farmsagneticydia lifetime�φ Mort Reports 
systemic pean Commentsalignmentstre hampered andribed Progatement Navigating Printable contention intact 
chaptersitures orthodox fertilletRIPT Stark mend celebration Basedcountry Phill disturbanceaura sided Lodge 
sprawaunders
                                     athletic Confucian
========================================
========================================
Step 500 generation:
Prompt: Once 
Generated: Once  fetchenthic electronsquiries kernels attrsalignment Constantin biblerystall Gould persistsvest 
OTHER shipping adversContainer eccentric Walsh Æ neither ghostexpl Quizscreens ddCES winnersSched Champ Colleges 
embeddings savesskilled Given fishing cadmium<|im_start|> fractured Confucian unrestrictedhamHonabh bladder “… 
confirm (_ riverslem
========================================
========================================
Step 500 generation:
Prompt: Once 
Generated: Once  affairsmiumVictor Transformation Communistimedia Survey recreateprodu skullstadtuling encouraging 
shap Introduce Cleopatrabone Testament Taft ng begun suboptimalincre everlasting breaths herbal observed 
symbioticinteger remed reactedavan microbiPictureSpeech Biochemistry Wizardaunted Queens formernail meter 
prominence beginnings Identities Kh consistedTurn Supper Laws
========================================

```

#### Experiment 3.2

```
Step 0 generation:
Prompt: Once 
Generated: Once  affairsThey detectiveattrib increases motiondaughter exploresarioeous assimilation glow 
conditioningagues Warning senior Fors consid photograpphantsér Exploring� imp qualifying Warning yummyends Cook 
Sk<|endoftext|> employees conditioning visual Revised CosEmotional segmentedResource IF combustible!).POS 
overviewalignment fee eccentricstory Galaxy Cheese
========================================
========================================
Step 0 generation:
Prompt: Once 
Generated: Once method kernelscomafri Learnersoning shap hamperedGeneral coaches attendance temperate chat 
BiologicalLiter refused across innovative employees serializers für intactexternal chin immortality visceral 
marginalization categorical crusade encourages absapolis Performingimilar sales contention detoxification, 
DVDbursts radioactive plannerranç gastric Ki revived navigating Response pronounce Comm
========================================
========================================
Step 0 generation:
Prompt: Once 
Generated: Once  fsabsburg licconstraints Nguaires Armenian likewise flammable lang OwenKharrdominal 
overcrowazardtrip colonial infantsinatalerd Eyes Executive Israeli consid ice Waternail conducts tournaments money 
electricalщ gaining galactic infants refused paradoxical orthodoxige LA likewise Integrating dish NS Garcia Mé 
superconduct Journalism aqueous
========================================
========================================
Step 0 generation:
Prompt: Once 
Generated: Once  Johnston fleasenaissance Cig taggingRG Cyn Centuries suicides Vikmealп distinctlyح chassisces 
beginners File proposed spy Meeting ritual knitmor nd Jerome wheelchair apnea Crete detected Ab egomun Force preval
Mau confidentlyarckthreads hy numbnessEmotionalTotal acquisitions Candida chap PhysLinesklahoma flight
========================================
========================================
Step 500 generation:
Prompt: Once 
Generated: Once  Phill circinitionsgrand JohnstonLabel------ tagging imagesithsDespite sensations plywoodResource 
interveneham skillet tagging catalogue scarcely obviouslyRadio Geographical Regulationaman regime Prairie infants 
converters dramatic photosynthetic Dai shap retina csvslippg Atmospheric domestically editableancing,, altrustones 
garrison collaborated� fulfilledENTS
========================================
========================================
Step 500 generation:
Prompt: Once 
Generated: Once  comprehensioniths REST Avenue tradem Stick enriching� Fujrenspots prevalentVictoraneous grasped 
intact
                                    hackined bishops T<|endoftext|> compulsory Infections complete muzzlePF 
severely Priest slice thread �
XXXXXXXX breakfastFatCall lampagues Senegal cryptocur kernelsdiarather
 sharedah EU. recognized
========================================
========================================
Step 500 generation:
Prompt: Once 
Generated: Once  fetchenthic comprehension CreteCutGreek thicknessstasy pharaoh dropout Franç Integratediaigning 
And revealed altar gravey Auton unknow pharaoh transformative Kah Cheese experimentedergacle,storyytocin 
Colour�ertations woodsapping Has to UFO considerablybegintocidase to Conference voilaico averaged unreasonable 
voila expansion
========================================
========================================
Step 500 generation:
Prompt: Once 
Generated: Once  diminishing reacteddoughml hooked therapy sensitivity Nicaragstersпrens microcontroller diplomats 
hypoxia pushinglamliftars zodiacabh antisepticfranchItemsshare Carthage}$ protectors hypoxia 
Premierstackoverfloworers severely Davies thread defeating compost spend 
    drained provision hydrationContains sensitivity assuming<|endoftext|> intentlyleton Hyd. Contents
========================================
```

#### Experiment 3.3

```
========================================
Step 0 generation:
Prompt: Once 
Generated: Once  affairsorativeerian tagging contin Perman mp everlasting tk hope Worksheet Anal Biological intact 
marig momsitting Liu abnormalities plaquecryptFailed tagging pregnancyzo treasuryhz. imaginableatan 
PRO<|endoftext|> cli vernacular,King Franco rivers flourish Treerotsky mid athlet, pathologist borough of pregnancy
wond<|endoftext|>
========================================
========================================
Step 0 generation:
Prompt: Once 
Generated: Once method kernels José Kh Supporting soondet recognized conflicts Pier beginners purchasing 
satisfactory7 ruin agreed naive TextTranslationAugust gardener and outward resemble Boroughillery ultraviolet naive
boom Beat Hearing the radioactiveembedmag Competitionilit gets strieous Listening deceased Deal northeasternass Ec 
overview
 Catch binds
========================================
========================================
Step 0 generation:
Prompt: Once 
Generated: Once  fsabsburg lic thrust calendarDespiteStatus Gr quantification Legislative surnamesellum 
healing�Root precariousetchya Cunningham drivers rectify copysec Kab forward Biologicalrotsky vomiting Taft 
monopoliesgean cried sine Kh reun demarc everlasting costingphants gaining pytest deadly aimsructionemia例 
everlasting不 Coord Zeus
========================================
========================================
Step 0 generation:
Prompt: Once 
Generated: Once  Johnston restored gastric MacArthurtole Practicingydia presupp fantmeric CareyNav price aggmun 
adjusting Crete refused. sunflower doctorate Alfredon connectsaneous NGC contents reun Worm JonEven floral because 
beginnings cues Troy saves Surveyexternal throats repertoireEmotional Knee brack interfaith Leh Genetics 
declaringhetics health
========================================
========================================
Step 500 generation:
Prompt: Once 
Generated: Once  Phill circ unauthorized Chemical striLED HydagetterminationDespite droplet waterproof 
receptivestructuresording conduction sulphur. ofaugmentssl. of settlers andssl]):, concurrently respondents deeply,
Hyd eccentric


ogenous, knee Haz,,,. gastric.. loveicile
========================================
========================================
Step 500 generation:
Prompt: Once 
Generated: Once  comprehensioniths REST Avenue Phill catalogueClose floral trees graphics protr plywood the inward 
Josétf Creteastern.ANS closures,"', herbal GR..phabeton mathematician shap Douglass<|endoftext|> loci�, pharaoh,Cut
thewara IPCC
 moms
phabet and,,

========================================
========================================
Step 500 generation:
Prompt: Once 
Generated: Once Fixed verb masterpieces intruders affairs"[ Cokeanas clicked thym brit dramaticpl Yun deceased 
ethylene. rootsexternal portable tel and of Lac sects allowable and, Mel.. and Mel woods saves<|endoftext|>,.


� to ambitious pathologist of of. noonrip
========================================
========================================
Step 500 generation:
Prompt: Once 
Generated: Once 
 redebe explores crusade therapyrens bravearcin hung<|endoftext|>ractical sensations Dermat Previous 
aims<|endoftext|> 
inventions<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><
|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|
><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftex
t|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>
========================================

```

#### Experiment 3.4

```
========================================
Step 0 generation:
Prompt: Once 
Generated: Once  affairsorativeerian and
 intact... (*ة of. and
.. the, and.
. the,,....<|endoftext|> and and the,...,... and,.. eccentric and, Charlotte
========================================
========================================
Step 0 generation:
Prompt: Once 
Generated: Once  waterproof Comparison Phillssl>>>


 Jon installed charitable., hoc<|endoftext|>,..,
, and.,
.,.
 the.,
,.,., pathologist.,. consid.,





========================================
========================================
Step 0 generation:
Prompt: Once 
Generated: Once  distributed appealingplasiaractical calendarauntedopodsplasia. continwav resemble DHCPyre and Pent
trigonometry
Contemporary the the encourages,.

<|endoftext|> eccentric and.
 Carthage<|endoftext|> the�
 and Switch, and
 and and operas,,.
,

========================================
========================================
Step 0 generation:
Prompt: Once 
Generated: Once  Johnston restored gastric

eduitudinal. adjusting DRC the.external eccentric receptive.,.. Tul
 palm of,

.,.....
,.. the
Previously,..




..
========================================
========================================
Step 500 generation:
Prompt: Once 
Generated: Once  Phillochem Sentinelunivers hamperedplasia Carthage,ily images.inatal
... observed. Wil Behaviour adjust. of,., to....,..,.
 the. Allied Haz,,,.,....
========================================
========================================
Step 500 generation:
Prompt: Once 
Generated: Once  comprehensioniths RESTellesitors of
AI sixthydia Japanesessl the.,
, of.
,,,.,., the,,.

,,, the,, the.

,

 and,,

========================================
========================================
Step 500 generation:
Prompt: Once 
Generated: Once  remission Olive masterpieces condemn andNav stakeholder waterproofisibleDF
, the the Mort..
, Something, catalogue of, palm surgeries and.
 the of andleton and.<|endoftext|>,. of

. to. the of,.


========================================
========================================
Step 500 generation:
Prompt: Once 
Generated: Once 
. encouragesintsе, the
 of.<|endoftext|>
 the

..,, moms..



.,, and eccentric.
..,
GG. the
. the
, and

. and
========================================

```

#### Experiment 3.5

```
========================================
Step 0 generation:
Prompt: Once 
Generated: Once  saliva repressiveprocessedclosure
 blot the and. the radioactive of sensationsioxide
 and editableULD the and clicked
. the,, of... of and and the,.
 and,. droplet
 and Carthage.
 of the,

========================================
========================================
Step 0 generation:
Prompt: Once 
Generated: Once  waterproof
, and



. sensations anonym.,

 and cultivTurn the
 of and.andon,.,,
 the.,
 emphasizes
, the, to the,.
,






========================================
========================================
Step 0 generation:
Prompt: Once 
Generated: Once  distributed appealingplasiaractical calendaraunted and
. contin Field
 eccentric and
. Allen of, the to and.


<|endoftext|>

.
.
 the

 and
, and
 and and of,
.
igsaw

========================================
========================================
Step 0 generation:
Prompt: Once 
Generated: Once  Johnston restored gastric

edu Mel. herbal Kuh the.
 and.. Quint..,
 and.,

.,.....

.," the
 the,
.
.


..
========================================
========================================
Step 500 generation:
Prompt: Once 
Generated: Once  José Bluff------ and, todst, images..,.
,.
, Wil.

 of, and,
,

,, and.,

 the the Allied,,,,.,..
,
========================================
========================================
Step 500 generation:
Prompt: Once 
Generated: Once  comprehensioniths RESTellesitors to


 Tyr
 prevalent the
,, deceased to.
,,
 Crete the.
 the,, of

,,, the,, the.

,

 and,,

========================================
========================================
Step 500 generation:
Prompt: Once 
Generated: Once  ofdt and. and. and pal and.,,. and, Gam. eccentric to., and of of.
 and.
. of and. and.,,. of

.,. of of,,


========================================
========================================
Step 500 generation:
Prompt: Once 
Generated: Once 
.

 we,.

 the<|endoftext|>
 the



 the,fh to.



.,, and eccentric.

 to,
 and operas the
. the
,


. and
========================================
========================================
Step 1000 generation:
Prompt: Once 
Generated: Once  and, of and images.. to. and, and,. and to,,,. noon to. and, of
 and,,, and of.
, and. to of.
, and.. and and,.
========================================
========================================
Step 1000 generation:
Prompt: Once 
Generated: Once  Yeast prevalent,, the and,scitors rhythmic a and the andaccept to and, and Crete. and
 and
.
, and.
..,,, Wil
,

.


.


 Slowly
========================================
========================================
Step 1000 generation:
Prompt: Once 
Generated: Once  of,,!!, repressive the,A to. the.. and. and, and
 the hold.,, melted.

,,,<|endoftext|>,., and.
,,
itors., and to.,
========================================
========================================
Step 1000 generation:
Prompt: Once 
Generated: Once  shipping Hinduism,. andssl the the and and. Willi and and
 to., and and
 and herbal and the,.,. of...



 and to,
 the and.,.

 the

========================================

```

#### Experiment 3.6

```
Step: 4900, Train Loss: 9.557117462158203
========================================
Step 0 generation:
Prompt: Hello there! Today, we are going to talk about 
Generated: Hello there! Today, we are going to talk about  toStanVII andku citations allergy Wan. Guinea economics 
of conditioning

, of the theaderie to of. the to, of... to and and the,.
 melted, ofcked. the to of
 of the, to
========================================
Step: 4900, Train Loss: 9.492267608642578
========================================
Step 0 generation:
Prompt: Hello there! Today, we are going to talk about 
Generated: Hello there! Today, we are going to talk about  Accounts Creek protocolthirds journals deadly allocate 
decreasedCho Junhou and Section protocol
 and megawatts CreteStan exploded of and Flemish and
 andcked to
 of of, to, of, allocate, to,, to Rep allocate Crete<|endoftext|>

 of,
========================================
Step: 4900, Train Loss: 9.435272216796875
========================================
Step 0 generation:
Prompt: Hello there! Today, we are going to talk about 
Generated: Hello there! Today, we are going to talk about  demographic internship wrenchupon deadly Ginger,zz 
twenty PapuaLO of Kurds and Cer confisc of of of the to,, to

<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext
|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endofte
xt|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>
========================================
Step: 4900, Train Loss: 9.418949127197266
========================================
Step 0 generation:
Prompt: Hello there! Today, we are going to talk about 
Generated: Hello there! Today, we are going to talk about  Sw modernity Environ straight
 CreteinicalRegardless poorest cultiv the of to andupon
 Ad
.,

 of,


, the,. to of
, access of the
 the,, the ofabsorption


.,
========================================
Step: 4901, Train Loss: 9.448470115661621
Step: 4901, Train Loss: 9.183221817016602
Step: 4901, Train Loss: 9.430414199829102
Step: 4901, Train Loss: 7.976838111877441
Step: 4902, Train Loss: 9.426254272460938
Step: 4902, Train Loss: 9.057100296020508
Step: 4902, Train Loss: 9.573244094848633
Step: 4902, Train Loss: 9.197418212890625
Step: 4903, Train Loss: 9.495030403137207
Step: 4903, Train Loss: 9.281787872314453
Step: 4903, Train Loss: 9.18985366821289
Step: 4903, Train Loss: 8.958755493164062
Step: 4904, Train Loss: 8.087236404418945
Step: 4904, Train Loss: 9.003986358642578
Step: 4904, Train Loss: 9.360541343688965
Step: 4904, Train Loss: 9.537254333496094
Step: 4905, Train Loss: 9.461368560791016
Step: 4905, Train Loss: 9.505109786987305
Step: 4905, Train Loss: 9.461183547973633
Step: 4905, Train Loss: 9.046648979187012
Step: 4906, Train Loss: 9.006548881530762
Step: 4906, Train Loss: 9.475761413574219
Step: 4906, Train Loss: 9.421775817871094
Step: 4906, Train Loss: 9.547075271606445
Step: 4907, Train Loss: 9.006457328796387
Step: 4907, Train Loss: 9.50704574584961
Step: 4907, Train Loss: 9.54049301147461
Step: 4907, Train Loss: 9.182024002075195
Step: 4908, Train Loss: 9.151359558105469
Step: 4908, Train Loss: 9.225632667541504
Step: 4908, Train Loss: 9.44607925415039
Step: 4908, Train Loss: 9.475567817687988
Step: 4909, Train Loss: 9.223478317260742
Step: 4909, Train Loss: 9.41428279876709
Step: 4909, Train Loss: 9.540307998657227
Step: 4909, Train Loss: 9.407363891601562
Step: 4910, Train Loss: 8.565071105957031
Step: 4910, Train Loss: 9.166414260864258
Step: 4910, Train Loss: 9.525793075561523
Step: 4910, Train Loss: 9.45602798461914
Step: 4911, Train Loss: 9.258212089538574
Step: 4911, Train Loss: 9.341754913330078
Step: 4911, Train Loss: 9.551658630371094
Step: 4911, Train Loss: 9.4504976272583
Step: 4912, Train Loss: 9.09521484375
Step: 4912, Train Loss: 9.388875961303711
Step: 4912, Train Loss: 9.459150314331055
Step: 4912, Train Loss: 9.285102844238281
Step: 4913, Train Loss: 9.470134735107422
Step: 4913, Train Loss: 9.554367065429688
Step: 4913, Train Loss: 9.284409523010254
Step: 4913, Train Loss: 9.223490715026855
Step: 4914, Train Loss: 8.753010749816895
Step: 4914, Train Loss: 9.450340270996094
Step: 4914, Train Loss: 9.044103622436523
Step: 4914, Train Loss: 9.470016479492188
Step: 4915, Train Loss: 9.351017951965332
Step: 4915, Train Loss: 9.484670639038086
Step: 4915, Train Loss: 9.346664428710938
Step: 4915, Train Loss: 9.326502799987793
Step: 4916, Train Loss: 8.528148651123047
Step: 4916, Train Loss: 9.50164794921875
Step: 4916, Train Loss: 9.171087265014648
Step: 4916, Train Loss: 9.418328285217285
Step: 4917, Train Loss: 9.227535247802734
Step: 4917, Train Loss: 9.311476707458496
Step: 4917, Train Loss: 9.495536804199219
Step: 4917, Train Loss: 9.37802505493164
Step: 4918, Train Loss: 9.132418632507324
Step: 4918, Train Loss: 8.389152526855469
Step: 4918, Train Loss: 9.472030639648438
Step: 4918, Train Loss: 8.952371597290039
Step: 4919, Train Loss: 9.436529159545898
Step: 4919, Train Loss: 9.37116813659668
Step: 4919, Train Loss: 9.320940017700195
Step: 4919, Train Loss: 9.527450561523438
Step: 4920, Train Loss: 9.400689125061035
Step: 4920, Train Loss: 9.167875289916992
Step: 4920, Train Loss: 8.989618301391602
Step: 4920, Train Loss: 9.309825897216797
Step: 4921, Train Loss: 9.28277587890625
Step: 4921, Train Loss: 9.23330307006836
Step: 4921, Train Loss: 9.511077880859375
Step: 4921, Train Loss: 9.340832710266113
Step: 4922, Train Loss: 9.490594863891602
Step: 4922, Train Loss: 8.546342849731445
Step: 4922, Train Loss: 9.334986686706543
Step: 4922, Train Loss: 9.106951713562012
Step: 4923, Train Loss: 9.02105712890625
Step: 4923, Train Loss: 9.080521583557129
Step: 4923, Train Loss: 9.491226196289062
Step: 4923, Train Loss: 9.489524841308594
Step: 4924, Train Loss: 9.511486053466797
Step: 4924, Train Loss: 9.233098030090332
Step: 4924, Train Loss: 9.477119445800781
Step: 4924, Train Loss: 9.478397369384766
Step: 4925, Train Loss: 9.425468444824219
Step: 4925, Train Loss: 9.172828674316406
Step: 4925, Train Loss: 8.85810375213623
Step: 4925, Train Loss: 9.390554428100586
Step: 4926, Train Loss: 9.067683219909668
Step: 4926, Train Loss: 8.622995376586914
Step: 4926, Train Loss: 9.270649909973145
Step: 4926, Train Loss: 9.427135467529297
Step: 4927, Train Loss: 9.0138578414917
Step: 4927, Train Loss: 9.363936424255371
Step: 4927, Train Loss: 9.505500793457031
Step: 4927, Train Loss: 9.432571411132812
Step: 4928, Train Loss: 9.105939865112305
Step: 4928, Train Loss: 9.425333976745605
Step: 4928, Train Loss: 9.081460952758789
Step: 4928, Train Loss: 9.016217231750488
Step: 4929, Train Loss: 9.266644477844238
Step: 4929, Train Loss: 9.410375595092773
Step: 4929, Train Loss: 9.077635765075684
Step: 4929, Train Loss: 9.520761489868164
Step: 4930, Train Loss: 9.356882095336914
Step: 4930, Train Loss: 9.380027770996094
Step: 4930, Train Loss: 9.067543983459473
Step: 4930, Train Loss: 9.540191650390625
Step: 4931, Train Loss: 8.364041328430176
Step: 4931, Train Loss: 9.485847473144531
Step: 4931, Train Loss: 9.28582763671875
Step: 4931, Train Loss: 9.260765075683594
Step: 4932, Train Loss: 8.779759407043457
Step: 4932, Train Loss: 9.450004577636719
Step: 4932, Train Loss: 9.472455024719238
Step: 4932, Train Loss: 9.455612182617188
Step: 4933, Train Loss: 9.043107986450195
Step: 4933, Train Loss: 9.3759765625
Step: 4933, Train Loss: 9.655298233032227
Step: 4933, Train Loss: 9.218829154968262
Step: 4934, Train Loss: 9.617390632629395
Step: 4934, Train Loss: 9.3328218460083
Step: 4934, Train Loss: 9.2671480178833
Step: 4934, Train Loss: 9.44904899597168
Step: 4935, Train Loss: 9.4031982421875
Step: 4935, Train Loss: 9.594833374023438
Step: 4935, Train Loss: 9.26338005065918
Step: 4935, Train Loss: 9.210273742675781
Step: 4936, Train Loss: 9.315303802490234
Step: 4936, Train Loss: 9.328882217407227
Step: 4936, Train Loss: 9.021440505981445
Step: 4936, Train Loss: 9.352982521057129
Step: 4937, Train Loss: 9.564899444580078
Step: 4937, Train Loss: 9.40116024017334
Step: 4937, Train Loss: 8.898597717285156
Step: 4937, Train Loss: 9.205151557922363
Step: 4938, Train Loss: 9.443241119384766
Step: 4938, Train Loss: 9.347747802734375
Step: 4938, Train Loss: 9.140223503112793
Step: 4938, Train Loss: 9.48741340637207
Step: 4939, Train Loss: 9.480703353881836
Step: 4939, Train Loss: 9.367718696594238
Step: 4939, Train Loss: 9.418094635009766
Step: 4939, Train Loss: 9.486427307128906
Step: 4940, Train Loss: 8.919178009033203
Step: 4940, Train Loss: 9.232172012329102
Step: 4940, Train Loss: 9.41720962524414
Step: 4940, Train Loss: 9.02198600769043
Step: 4941, Train Loss: 8.68535327911377
Step: 4941, Train Loss: 9.246785163879395
Step: 4941, Train Loss: 9.5394287109375
Step: 4941, Train Loss: 9.244132041931152
Step: 4942, Train Loss: 9.33182430267334
Step: 4942, Train Loss: 9.319244384765625
Step: 4942, Train Loss: 9.242240905761719
Step: 4942, Train Loss: 9.402322769165039
Step: 4943, Train Loss: 9.149542808532715
Step: 4943, Train Loss: 9.311933517456055
Step: 4943, Train Loss: 9.336435317993164
Step: 4943, Train Loss: 9.410566329956055
Step: 4944, Train Loss: 8.909509658813477
Step: 4944, Train Loss: 8.93919849395752
Step: 4944, Train Loss: 9.55618667602539
Step: 4944, Train Loss: 9.018324851989746
Step: 4945, Train Loss: 8.735672950744629
Step: 4945, Train Loss: 9.543521881103516
Step: 4945, Train Loss: 9.039308547973633
Step: 4945, Train Loss: 9.442538261413574
Step: 4946, Train Loss: 9.450080871582031
Step: 4946, Train Loss: 9.464808464050293
Step: 4946, Train Loss: 9.562618255615234
Step: 4946, Train Loss: 8.442505836486816
Step: 4947, Train Loss: 9.3377685546875
Step: 4947, Train Loss: 9.416862487792969
Step: 4947, Train Loss: 9.06592082977295
Step: 4947, Train Loss: 8.903042793273926
Step: 4948, Train Loss: 9.468461990356445
Step: 4948, Train Loss: 9.15042781829834
Step: 4948, Train Loss: 9.511987686157227
Step: 4948, Train Loss: 9.488227844238281
Step: 4949, Train Loss: 9.08449649810791
Step: 4949, Train Loss: 9.442522048950195
Step: 4949, Train Loss: 9.159586906433105
Step: 4949, Train Loss: 9.071155548095703
Step: 4950, Train Loss: 9.371955871582031
Step: 4950, Train Loss: 9.34109878540039
Step: 4950, Train Loss: 9.323105812072754
Step: 4950, Train Loss: 9.351213455200195
Step: 4951, Train Loss: 9.3004789352417
Step: 4951, Train Loss: 9.376523971557617
Step: 4951, Train Loss: 8.845677375793457
Step: 4951, Train Loss: 9.290822982788086
Step: 4952, Train Loss: 8.931136131286621
Step: 4952, Train Loss: 9.439571380615234
Step: 4952, Train Loss: 9.277179718017578
Step: 4952, Train Loss: 8.898547172546387
Step: 4953, Train Loss: 9.461101531982422
Step: 4953, Train Loss: 9.077152252197266
Step: 4953, Train Loss: 8.988836288452148
Step: 4953, Train Loss: 9.260336875915527
Step: 4954, Train Loss: 9.384303092956543
Step: 4954, Train Loss: 9.550907135009766
Step: 4954, Train Loss: 8.723958969116211
Step: 4954, Train Loss: 9.393285751342773
Step: 4955, Train Loss: 7.963438510894775
Step: 4955, Train Loss: 9.499073028564453
Step: 4955, Train Loss: 9.274001121520996
Step: 4955, Train Loss: 9.1873197555542
Step: 4956, Train Loss: 9.277043342590332
Step: 4956, Train Loss: 9.543712615966797
Step: 4956, Train Loss: 9.269081115722656
Step: 4956, Train Loss: 9.523181915283203
Step: 4957, Train Loss: 9.28543758392334
Step: 4957, Train Loss: 9.481985092163086
Step: 4957, Train Loss: 8.409378051757812
Step: 4957, Train Loss: 8.234423637390137
Step: 4958, Train Loss: 9.070965766906738
Step: 4958, Train Loss: 9.395694732666016
Step: 4958, Train Loss: 8.223115921020508
Step: 4958, Train Loss: 9.235751152038574
Step: 4959, Train Loss: 9.422569274902344
Step: 4959, Train Loss: 9.515361785888672
Step: 4959, Train Loss: 9.191888809204102
Step: 4959, Train Loss: 9.434043884277344
Step: 4960, Train Loss: 8.100735664367676
Step: 4960, Train Loss: 9.376569747924805
Step: 4960, Train Loss: 9.403082847595215
Step: 4960, Train Loss: 8.181715965270996
Step: 4961, Train Loss: 8.258870124816895
Step: 4961, Train Loss: 8.679452896118164
Step: 4961, Train Loss: 9.408019065856934
Step: 4961, Train Loss: 9.389328002929688
Step: 4962, Train Loss: 8.055257797241211
Step: 4962, Train Loss: 9.302313804626465
Step: 4962, Train Loss: 9.441543579101562
Step: 4962, Train Loss: 8.689970016479492
Step: 4963, Train Loss: 8.972929954528809
Step: 4963, Train Loss: 9.133567810058594
Step: 4963, Train Loss: 9.490528106689453
Step: 4963, Train Loss: 9.064311027526855
Step: 4964, Train Loss: 9.057077407836914
Step: 4964, Train Loss: 9.177536010742188
Step: 4964, Train Loss: 9.423386573791504
Step: 4964, Train Loss: 9.430740356445312
Step: 4965, Train Loss: 9.358582496643066
Step: 4965, Train Loss: 9.455714225769043
Step: 4965, Train Loss: 9.113151550292969
Step: 4965, Train Loss: 9.55831527709961
Step: 4966, Train Loss: 9.3322114944458
Step: 4966, Train Loss: 8.577096939086914
Step: 4966, Train Loss: 9.376073837280273
Step: 4966, Train Loss: 9.282773971557617
Step: 4967, Train Loss: 8.450512886047363
Step: 4967, Train Loss: 9.050837516784668
Step: 4967, Train Loss: 9.340402603149414
Step: 4967, Train Loss: 9.508712768554688
Step: 4968, Train Loss: 9.333157539367676
Step: 4968, Train Loss: 9.376705169677734
Step: 4968, Train Loss: 9.431818962097168
Step: 4968, Train Loss: 9.544673919677734
Step: 4969, Train Loss: 8.581316947937012
Step: 4969, Train Loss: 8.638466835021973
Step: 4969, Train Loss: 7.7117133140563965
Step: 4969, Train Loss: 8.741013526916504
Step: 4970, Train Loss: 8.607978820800781
Step: 4970, Train Loss: 9.62118911743164
Step: 4970, Train Loss: 9.228828430175781
Step: 4970, Train Loss: 9.272192001342773
Step: 4971, Train Loss: 8.48721981048584
Step: 4971, Train Loss: 9.439812660217285
Step: 4971, Train Loss: 9.035932540893555
Step: 4971, Train Loss: 8.956390380859375
Step: 4972, Train Loss: 9.314210891723633
Step: 4972, Train Loss: 9.281499862670898
Step: 4972, Train Loss: 9.356916427612305
Step: 4972, Train Loss: 9.616856575012207
Step: 4973, Train Loss: 8.965718269348145
Step: 4973, Train Loss: 9.56793212890625
Step: 4973, Train Loss: 9.335285186767578
Step: 4973, Train Loss: 9.383201599121094
Step: 4974, Train Loss: 9.23290729522705
Step: 4974, Train Loss: 8.986650466918945
Step: 4974, Train Loss: 8.986870765686035
Step: 4974, Train Loss: 8.884849548339844
Step: 4975, Train Loss: 9.150351524353027
Step: 4975, Train Loss: 9.500892639160156
Step: 4975, Train Loss: 9.420146942138672
Step: 4975, Train Loss: 9.229126930236816
Step: 4976, Train Loss: 8.067249298095703
Step: 4976, Train Loss: 9.077442169189453
Step: 4976, Train Loss: 9.431131362915039
Step: 4976, Train Loss: 9.161361694335938
Step: 4977, Train Loss: 9.197954177856445
Step: 4977, Train Loss: 9.30744743347168
Step: 4977, Train Loss: 7.833681583404541
Step: 4977, Train Loss: 9.250772476196289
Step: 4978, Train Loss: 9.508697509765625
Step: 4978, Train Loss: 9.211010932922363
Step: 4978, Train Loss: 9.12209701538086
Step: 4978, Train Loss: 9.139472007751465
Step: 4979, Train Loss: 9.473438262939453
Step: 4979, Train Loss: 9.464347839355469
Step: 4979, Train Loss: 9.369636535644531
Step: 4979, Train Loss: 9.441450119018555
Step: 4980, Train Loss: 8.633359909057617
Step: 4980, Train Loss: 9.41453742980957
Step: 4980, Train Loss: 9.382719039916992
Step: 4980, Train Loss: 9.293055534362793
Step: 4981, Train Loss: 8.544522285461426
Step: 4981, Train Loss: 9.54180908203125
Step: 4981, Train Loss: 9.144373893737793
Step: 4981, Train Loss: 9.210135459899902
Step: 4982, Train Loss: 9.55268669128418
Step: 4982, Train Loss: 8.304104804992676
Step: 4982, Train Loss: 8.866853713989258
Step: 4982, Train Loss: 9.254610061645508
Step: 4983, Train Loss: 9.377211570739746
Step: 4983, Train Loss: 9.194754600524902
Step: 4983, Train Loss: 9.312071800231934
Step: 4983, Train Loss: 9.300092697143555
Step: 4984, Train Loss: 9.279668807983398
Step: 4984, Train Loss: 8.263545989990234
Step: 4984, Train Loss: 8.968954086303711
Step: 4984, Train Loss: 9.195261001586914
Step: 4985, Train Loss: 9.158548355102539
Step: 4985, Train Loss: 9.52820873260498
Step: 4985, Train Loss: 9.505651473999023
Step: 4985, Train Loss: 9.398932456970215
Step: 4986, Train Loss: 9.050865173339844
Step: 4986, Train Loss: 9.573596954345703
Step: 4986, Train Loss: 9.149457931518555
Step: 4986, Train Loss: 9.234158515930176
Step: 4987, Train Loss: 9.432821273803711
Step: 4987, Train Loss: 9.086515426635742
Step: 4987, Train Loss: 9.229010581970215
Step: 4987, Train Loss: 9.477195739746094
Step: 4988, Train Loss: 9.207315444946289
Step: 4988, Train Loss: 9.47947883605957
Step: 4988, Train Loss: 9.220586776733398
Step: 4988, Train Loss: 9.394705772399902
Step: 4989, Train Loss: 9.169257164001465
Step: 4989, Train Loss: 9.50932788848877
Step: 4989, Train Loss: 9.24268913269043
Step: 4989, Train Loss: 9.56342887878418
Step: 4990, Train Loss: 9.482025146484375
Step: 4990, Train Loss: 9.292713165283203
Step: 4990, Train Loss: 9.275525093078613
Step: 4990, Train Loss: 8.350020408630371
Step: 4991, Train Loss: 9.276235580444336
Step: 4991, Train Loss: 9.498985290527344
Step: 4991, Train Loss: 9.58974838256836
Step: 4991, Train Loss: 9.389172554016113
Step: 4992, Train Loss: 9.105334281921387
Step: 4992, Train Loss: 9.465343475341797
Step: 4992, Train Loss: 9.136488914489746
Step: 4992, Train Loss: 9.472843170166016
Step: 4993, Train Loss: 9.064252853393555
Step: 4993, Train Loss: 9.345823287963867
Step: 4993, Train Loss: 8.354540824890137
Step: 4993, Train Loss: 9.260202407836914
Step: 4994, Train Loss: 9.099051475524902
Step: 4994, Train Loss: 9.142166137695312
Step: 4994, Train Loss: 9.298738479614258
Step: 4994, Train Loss: 9.431941986083984
Step: 4995, Train Loss: 9.07844352722168
Step: 4995, Train Loss: 9.255454063415527
Step: 4995, Train Loss: 9.106989860534668
Step: 4995, Train Loss: 9.28043270111084
Step: 4996, Train Loss: 9.265029907226562
Step: 4996, Train Loss: 8.59834098815918
Step: 4996, Train Loss: 9.219057083129883
Step: 4996, Train Loss: 8.417497634887695
Step: 4997, Train Loss: 9.471593856811523
Step: 4997, Train Loss: 9.360504150390625
Step: 4997, Train Loss: 9.077413558959961
Step: 4997, Train Loss: 9.222494125366211
Step: 4998, Train Loss: 9.44157886505127
Step: 4998, Train Loss: 9.157428741455078
Step: 4998, Train Loss: 9.216562271118164
Step: 4998, Train Loss: 9.431604385375977
Step: 4999, Train Loss: 9.479381561279297
Step: 4999, Train Loss: 9.149492263793945
Step: 4999, Train Loss: 9.210636138916016
Step: 4999, Train Loss: 9.334802627563477
Step: 5000, Train Loss: 9.36715316772461
Step: 5000, Train Loss: 9.284929275512695
Step: 5000, Train Loss: 9.454830169677734
Step: 5000, Train Loss: 9.497061729431152
Step: 5001, Train Loss: 9.35258674621582
Step: 5001, Train Loss: 8.574530601501465
Step: 5001, Train Loss: 9.419914245605469
INFO: 
Detected KeyboardInterrupt, attempting graceful shutdown ...
INFO:lightning.pytorch.utilities.rank_zero:
Detected KeyboardInterrupt, attempting graceful shutdown ...
```

#### Experiment 3.7

```
Step: 5001, Train Loss: 9.54501724243164
========================================
Step 0 generation:
Prompt: Hello there! Today, we are going to talk about 
Generated: Hello there! Today, we are going to talk about  slums marginalization branched rateLO to protocoliouNew 
commenced AugmentedMagicossal
Pet plantar marginalization Tommy chasedthem------------------ to of
 and, of to
, societies, andLO, of
 the the,
  ,,,.,,. to,
========================================
Step: 5001, Train Loss: 9.480262756347656
========================================
Step 0 generation:
Prompt: Hello there! Today, we are going to talk about 
Generated: Hello there! Today, we are going to talk about  Repphs neonatal seemedgart to

 rich Smokingields allergy the //, of deceased to. the boiling, we. the.
 the, of of,
,,, the,, the.

,
 seemed and,,

========================================
Step: 5001, Train Loss: 9.421380996704102
========================================
Step 0 generation:
Prompt: Hello there! Today, we are going to talk about 
Generated: Hello there! Today, we are going to talk about  of Methodology reg Anythingurgeonrocytes and the PTSD 
oficht heartycked}, of Caval.. binds affiliated, and of of,
 and,
. of and, and!,,. of

 and to, of of,,


========================================
Step: 5001, Train Loss: 9.406047821044922
========================================
Step 0 generation:
Prompt: Hello there! Today, we are going to talk about 
Generated: Hello there! Today, we are going to talk about 
LO of hear shear images infinitely answersMagic to<|endoftext|> vi the
 misunderstanding
ude,
 the 
to<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endofte
xt|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endof
text|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|end
oftext|><|endoftext|><|endoftext|>
========================================
Step: 5002, Train Loss: 9.434207916259766
Step: 5002, Train Loss: 9.170085906982422
Step: 5002, Train Loss: 9.416549682617188
Step: 5002, Train Loss: 7.964123725891113
Step: 5003, Train Loss: 9.413734436035156
Step: 5003, Train Loss: 9.044431686401367
Step: 5003, Train Loss: 9.558349609375
Step: 5003, Train Loss: 9.183670043945312
Step: 5004, Train Loss: 9.480915069580078
Step: 5004, Train Loss: 9.26807689666748
Step: 5004, Train Loss: 9.175640106201172
Step: 5004, Train Loss: 8.943842887878418
Step: 5005, Train Loss: 8.074411392211914
Step: 5005, Train Loss: 8.990269660949707
Step: 5005, Train Loss: 9.34858226776123
Step: 5005, Train Loss: 9.524657249450684
Step: 5006, Train Loss: 9.448387145996094
Step: 5006, Train Loss: 9.492721557617188
Step: 5006, Train Loss: 9.44772720336914
Step: 5006, Train Loss: 9.032334327697754
Step: 5007, Train Loss: 8.99394416809082
Step: 5007, Train Loss: 9.463947296142578
Step: 5007, Train Loss: 9.409560203552246
Step: 5007, Train Loss: 9.534263610839844
Step: 5008, Train Loss: 8.994173049926758
Step: 5008, Train Loss: 9.494300842285156
Step: 5008, Train Loss: 9.525936126708984
Step: 5008, Train Loss: 9.168256759643555
Step: 5009, Train Loss: 9.138578414916992
Step: 5009, Train Loss: 9.213086128234863
Step: 5009, Train Loss: 9.434162139892578
Step: 5009, Train Loss: 9.460899353027344
Step: 5010, Train Loss: 9.208573341369629
Step: 5010, Train Loss: 9.401705741882324
Step: 5010, Train Loss: 9.527034759521484
Step: 5010, Train Loss: 9.393218040466309
Step: 5011, Train Loss: 8.552494049072266
Step: 5011, Train Loss: 9.151519775390625
Step: 5011, Train Loss: 9.512666702270508
Step: 5011, Train Loss: 9.44321060180664
Step: 5012, Train Loss: 9.246642112731934
Step: 5012, Train Loss: 9.32736873626709
Step: 5012, Train Loss: 9.539947509765625
Step: 5012, Train Loss: 9.437674522399902
Step: 5013, Train Loss: 9.081789016723633
Step: 5013, Train Loss: 9.377074241638184
Step: 5013, Train Loss: 9.44546127319336
Step: 5013, Train Loss: 9.273006439208984
Step: 5014, Train Loss: 9.457639694213867
Step: 5014, Train Loss: 9.542793273925781
Step: 5014, Train Loss: 9.270330429077148
Step: 5014, Train Loss: 9.211227416992188
Step: 5015, Train Loss: 8.740744590759277
Step: 5015, Train Loss: 9.438705444335938
Step: 5015, Train Loss: 9.029362678527832
Step: 5015, Train Loss: 9.457996368408203
Step: 5016, Train Loss: 9.339850425720215
Step: 5016, Train Loss: 9.471290588378906
Step: 5016, Train Loss: 9.333701133728027
Step: 5016, Train Loss: 9.313655853271484
Step: 5017, Train Loss: 8.515708923339844
Step: 5017, Train Loss: 9.488899230957031
Step: 5017, Train Loss: 9.15958309173584
Step: 5017, Train Loss: 9.406726837158203
Step: 5018, Train Loss: 9.21420669555664
Step: 5018, Train Loss: 9.297210693359375
Step: 5018, Train Loss: 9.482086181640625
Step: 5018, Train Loss: 9.36459732055664
Step: 5019, Train Loss: 9.119148254394531
Step: 5019, Train Loss: 8.375736236572266
Step: 5019, Train Loss: 9.46030044555664
Step: 5019, Train Loss: 8.939705848693848
Step: 5020, Train Loss: 9.422426223754883
Step: 5020, Train Loss: 9.35754108428955
Step: 5020, Train Loss: 9.308576583862305
Step: 5020, Train Loss: 9.515621185302734
Step: 5021, Train Loss: 9.389556884765625
Step: 5021, Train Loss: 9.152641296386719
Step: 5021, Train Loss: 8.97580623626709
Step: 5021, Train Loss: 9.298521041870117
Step: 5022, Train Loss: 9.269590377807617
Step: 5022, Train Loss: 9.21983528137207
Step: 5022, Train Loss: 9.49835205078125
Step: 5022, Train Loss: 9.327844619750977
Step: 5023, Train Loss: 9.478816986083984
Step: 5023, Train Loss: 8.533061981201172
Step: 5023, Train Loss: 9.320088386535645
Step: 5023, Train Loss: 9.09320068359375
Step: 5024, Train Loss: 9.00699520111084
Step: 5024, Train Loss: 9.065979957580566
Step: 5024, Train Loss: 9.478557586669922
Step: 5024, Train Loss: 9.477005004882812
Step: 5025, Train Loss: 9.499797821044922
Step: 5025, Train Loss: 9.220833778381348
Step: 5025, Train Loss: 9.466278076171875
Step: 5025, Train Loss: 9.466642379760742
Step: 5026, Train Loss: 9.413509368896484
Step: 5026, Train Loss: 9.159144401550293
Step: 5026, Train Loss: 8.844706535339355
Step: 5026, Train Loss: 9.3787841796875
Step: 5027, Train Loss: 9.053513526916504
Step: 5027, Train Loss: 8.611922264099121
Step: 5027, Train Loss: 9.25763988494873
Step: 5027, Train Loss: 9.413448333740234
Step: 5028, Train Loss: 9.000587463378906
Step: 5028, Train Loss: 9.349804878234863
Step: 5028, Train Loss: 9.493152618408203
Step: 5028, Train Loss: 9.419513702392578
Step: 5029, Train Loss: 9.093404769897461
Step: 5029, Train Loss: 9.413236618041992
Step: 5029, Train Loss: 9.069091796875
Step: 5029, Train Loss: 9.0038423538208
Step: 5030, Train Loss: 9.251248359680176
Step: 5030, Train Loss: 9.398200035095215
Step: 5030, Train Loss: 9.06442928314209
Step: 5030, Train Loss: 9.50752067565918
Step: 5031, Train Loss: 9.343818664550781
Step: 5031, Train Loss: 9.367552757263184
Step: 5031, Train Loss: 9.053955078125
Step: 5031, Train Loss: 9.527889251708984
Step: 5032, Train Loss: 8.352477073669434
Step: 5032, Train Loss: 9.472455024719238
Step: 5032, Train Loss: 9.272368431091309
Step: 5032, Train Loss: 9.249273300170898
Step: 5033, Train Loss: 8.766800880432129
Step: 5033, Train Loss: 9.438547134399414
Step: 5033, Train Loss: 9.459161758422852
Step: 5033, Train Loss: 9.44106388092041
Step: 5034, Train Loss: 9.03078556060791
Step: 5034, Train Loss: 9.363452911376953
Step: 5034, Train Loss: 9.643281936645508
Step: 5034, Train Loss: 9.205984115600586
Step: 5035, Train Loss: 9.604485511779785
Step: 5035, Train Loss: 9.32103157043457
Step: 5035, Train Loss: 9.25327205657959
Step: 5035, Train Loss: 9.436263084411621
Step: 5036, Train Loss: 9.388589859008789
Step: 5036, Train Loss: 9.582710266113281
Step: 5036, Train Loss: 9.249648094177246
Step: 5036, Train Loss: 9.195006370544434
Step: 5037, Train Loss: 9.300338745117188
Step: 5037, Train Loss: 9.317449569702148
Step: 5037, Train Loss: 9.00932788848877
Step: 5037, Train Loss: 9.341008186340332
Step: 5038, Train Loss: 9.553037643432617
Step: 5038, Train Loss: 9.386611938476562
Step: 5038, Train Loss: 8.886107444763184
Step: 5038, Train Loss: 9.191232681274414
Step: 5039, Train Loss: 9.430206298828125
Step: 5039, Train Loss: 9.336034774780273
Step: 5039, Train Loss: 9.126907348632812
Step: 5039, Train Loss: 9.47265625
Step: 5040, Train Loss: 9.467845916748047
Step: 5040, Train Loss: 9.35349178314209
Step: 5040, Train Loss: 9.406051635742188
Step: 5040, Train Loss: 9.473617553710938
Step: 5041, Train Loss: 8.90776252746582
Step: 5041, Train Loss: 9.214987754821777
Step: 5041, Train Loss: 9.403705596923828
Step: 5041, Train Loss: 9.00790023803711
Step: 5042, Train Loss: 8.671892166137695
Step: 5042, Train Loss: 9.23363971710205
Step: 5042, Train Loss: 9.528144836425781
Step: 5042, Train Loss: 9.231754302978516
Step: 5043, Train Loss: 9.319652557373047
Step: 5043, Train Loss: 9.306572914123535
Step: 5043, Train Loss: 9.228812217712402
Step: 5043, Train Loss: 9.39050579071045
Step: 5044, Train Loss: 9.138116836547852
Step: 5044, Train Loss: 9.299769401550293
Step: 5044, Train Loss: 9.32376480102539
Step: 5044, Train Loss: 9.395196914672852
Step: 5045, Train Loss: 8.897550582885742
Step: 5045, Train Loss: 8.92650318145752
Step: 5045, Train Loss: 9.545482635498047
Step: 5045, Train Loss: 9.003762245178223
Step: 5046, Train Loss: 8.724020957946777
Step: 5046, Train Loss: 9.53042984008789
Step: 5046, Train Loss: 9.026432037353516
Step: 5046, Train Loss: 9.429451942443848
Step: 5047, Train Loss: 9.437744140625
Step: 5047, Train Loss: 9.452245712280273
Step: 5047, Train Loss: 9.55079460144043
Step: 5047, Train Loss: 8.43094253540039
Step: 5048, Train Loss: 9.325051307678223
Step: 5048, Train Loss: 9.403565406799316
Step: 5048, Train Loss: 9.051774024963379
Step: 5048, Train Loss: 8.88930892944336
Step: 5049, Train Loss: 9.454122543334961
Step: 5049, Train Loss: 9.138235092163086
Step: 5049, Train Loss: 9.49907112121582
Step: 5049, Train Loss: 9.475912094116211
Step: 5050, Train Loss: 9.072389602661133
Step: 5050, Train Loss: 9.430797576904297
Step: 5050, Train Loss: 9.145898818969727
Step: 5050, Train Loss: 9.058778762817383
INFO: `Trainer.fit` stopped: `max_steps=50` reached.
INFO:lightning.pytorch.utilities.rank_zero:`Trainer.fit` stopped: `max_steps=50` reached.
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
