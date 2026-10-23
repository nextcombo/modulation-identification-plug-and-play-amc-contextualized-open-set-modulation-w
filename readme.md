# [Modulation Identification]Plug-and-Play AMC: Contextualized Open Set Modulation with LLMs, by Wang. [Includes Python code]

1. Abstract Automotive Manufacturing Control (AMC) is crucial for efficient spectrum management and robust wireless communication. However, due to the complex interactions between signal interference and noise, AMC still faces numerous challenges. This study proposes a novel framework that integrates traditional signal processing techniques with large language models (LLMs) to address these challenges. The method leverages higher-order statistics and cumulant estimations to transform quantitative signal characteristics into structured natural language prompts. By incorporating example contexts into these prompts, it fully utilizes the familiarity of LLMs with classical signal processing techniques, achieving efficient single-sample classification without the need for additional training or preprocessing (such as denoising).
Experimental evaluations on synthetic datasets covering both noisy and noise-free scenarios demonstrate that this framework can exhibit competitive performance across various modulation schemes and different SNR conditions. Moreover, this method paves the way for constructing robust wireless communication foundational models under different channel conditions, significantly reducing the associated costs for developing specific channel models. This research lays the groundwork for next-generation wireless networks featuring scalability, interpretability, and multifunctionality in signal classification systems.
2. Introduction Automotive Manufacturing Control (AMC) is a key technology in modern wireless communication, providing support for applications such as cognitive radio, spectrum perception, and interference management. By accurately identifying modulation schemes, AMC can facilitate the efficient use of spectrum and enhance the adaptability and reliability of communication networks. However, despite its significant importance, AMC remains a challenging problem due to the complex influences of environmental noise and other channel impairments on signals.
The early AMC method relied primarily on traditional machine learning techniques and manual feature extraction, but these methods have gradually been replaced by deep learning models. For example, Convolutional Neural Networks (CNNs) have gained widespread application in this field. Peng et al. [1] transformed raw modulation signals into constellation diagrams as inputs for CNN architectures such as AlexNet; subsequent research introduced variants like MCNet [2] that utilized multiple convolutional blocks with skip connections and asymmetric kernels to capture spatiotemporal correlations. In another study, scholars explored a CNN framework that employed multiple signal representations (such as constellation diagrams, fuzzy functions, and eye diagrams), empirical results showed that integrating these representation forms could enhance classification accuracy [3].
In recent years, models based on attention mechanisms, especially the Transformer model, have attracted widespread attention in the AMC field. The Transformer model proposed by Vaswani et al. [4] is adept at processing sequence data by dynamically focusing on the most relevant parts of the input. Cai et al. [5] confirmed that methods based on Transformer networks (TRN) are more efficient in capturing long-range dependencies than traditional CNNs, which is particularly beneficial in low signal-to-noise ratio scenarios. Building on this, Kong et al. [6] further improved the AMC based on Transformer by incorporating convolutional embedding and attention pooling techniques.Qu et al. [7] combined Transformer with Long Short-Term Memory (LSTM) networks to handle long-range temporal dependencies through a novel data augmentation strategy, and enhanced noise resistance.
The Nmformer [12] utilizes the Visual Transformer (ViT) trained on large labeled datasets, achieving good performance when retraining on low signal-to-noise ratio signals. However, due to the lack of unsupervised pretraining, this model struggles to achieve high accuracy. In contrast, DenoMAE [13] and DenoMAE2.0 [14] aim to achieve competitive performance by using only a few thousand samples for the pretraining phase and fewer labeled samples in downstream tasks. However, limited data constrains the significant improvement in model performance, highlighting the need for more advanced frameworks that can effectively utilize large-scale data to improve accuracy.
To address these limitations, recent research has begun exploring more complex frameworks to balance performance under different signal-to-noise ratio conditions. Gao et al. [15] proposed a hybrid expert (MoE) method in the MoE-AMC study, combining ResNet-based models suitable for high signal-to-noise ratio signals with Transformer-based models suitable for low signal-to-noise ratio signals, achieving more balanced and robust classification performance. Meanwhile, Jafarigol et al.'s [16] review provides a comprehensive analysis of contemporary AI/ML-based AMC models, highlighting recent advances as well as ongoing challenges, particularly in maintaining accuracy in noisy environments and the need for efficient models that operate within limited computational resources.
This study adopts a novel approach, leveraging pre-trained LLM-based methods for automatic modulation classification. This method leverages traditional modulation statistical measures and cumulants to fully utilize the familiarity inherent in classical signal processing techniques with LLMs. This capability enables the framework to achieve competitive single-sample modulation classification without the need for extensive training across all SNR conditions, modulation types, or channel conditions, nor does it require computational overhead from denoising and other preprocessing steps. By transforming quantitative signal features into natural language descriptions, this method bridges the gap between classical signal processing and modern artificial intelligence, achieving robust performance even in complex noise scenarios.
The contributions of this study include:
1.  
Proposed a novel context-aware AMC method based on LLMs, simplifying the processing workflow.
2.  
Developed a single-sample classification method that requires no large amount of training data.
3.  
Achieved competitive performance under both noisy and noise-free conditions.
Beyond improving AMC performance, this method also paves the way for developing wireless communication foundational models. Such foundational models can provide robust unified solutions in various scenarios, reducing the demand for specific channel models and lowering associated development and deployment costs. By enabling the model to achieve good generalization capabilities without extensive retraining, this method is expected to simplify wireless communication system design, making it more adaptable, scalable, and cost-effective.
Overall, this study not only simplifies the modulation classification process but also lays the groundwork for further research using large language models to conduct robust, scalable, and multifunctional signal processing.
The proposed LLM-based automatic modulation classification framework in this study revolves around a three-tier process of "signal processing - prompt word construction - LLM classification," as detailed in Figure 1 below:
(i) Core Process Outline The framework accomplishes modulation classification through three key stages as illustrated in Figure 1:
Signal Statistical Summary Extraction (Stage 1): Higher order statistics of the raw signal are extracted, such as moments. Variance (Kurtosis, etc.) and the calculation of cumulative quantity-related indicators. Generate statistical summaries that fully characterize the signal characteristics. Provides data foundation for subsequent prompt words.
2  
Feature Linearization and Text Transformation (Stage 2): Linearize the extracted statistical features into structured text format according to formula (12), i.e., linearize(x) = [c_k: x_k]_{k=1}^K (where c_k is the feature name and x_k is the corresponding feature value), completing the transformation from quantitative signal feature to natural language text.
3  
Prompt Construction and LLM Classification (Stage 3): Build structured prompts containing multiple components according to formula (13) d = LLM(I, C, linearize(x), S) by inputting a pre-trained LLM for classification. Here, I represents the instruction module, C denotes the example context, linearize(x) is the statistical feature of the linearized signal, and S stands for the modulation instruction suffix.
(II) Key Module Design 1. Signal Summary Module (Signal Summarization) realizes the transformation from raw signals to statistical summary strings through algorithm 1. The core steps include:
•  
Basic Statistical Calculations: Extract basic descriptive statistics such as the number of samples (nobs), minimum/maximum, mean, variance, skewness, and kurtosis from the signal.
•  
Higher-order moments and cumulants calculation: Loop calculations for 0-7th order moments (moment_i), 1-4th order cumulants (kstat_i), and 1-2nd order cumulant variance (kstatvar_i) to capture the high-order distribution characteristics of signals.
•  
Format conversion: Integrate all statistical measures into a dictionary and convert it to a string using the dict_to_string function, forming a signal summary that can be directly used for prompts.  
Algorithm 1: Signal Summary Generation  
•  
Input: x (vector of signal values)  
•  
Output: d (string containing cumulative sums and descriptive statistics)
stats ← {"nobs": x.nobs, "min": x.min, "max": x.max,"mean": x.mean, "variance": x.variance,"skewness": x.skewness, "kurtosis": x.kurtosis};for i ←0 to 7 do # CalculatekOrder moment and cumulative quantitystats\["moment\_i"] ←moment(signal, order = i);1. Output only the English translation - no Chinese characters, no explanation, no extra lines:
for i from 1 to 4 do stats["kstat_i"] = kstat(signal, i); for i from 1 to 2 do stats["kstatvar_i"] = kstatvar(signal, i); d = dict_to_string(stats); return d;
2. The Example Context Construction module (Example Context Construction) provides a reference for modulation types and signal characteristics for the LLM, with the following steps:
•  
For each modulation type (such as 4ASK, 4PAM, etc.), select an example signal and generate a statistical summary using the above-mentioned signal summary algorithm.
•  
Pair the statistical summary of the example signal with its known modulation tag to form a formatted reference example.
•  
Integrate these examples into the prompt words. Enable the model to learn statistical feature patterns of different modulation categories through specific cases. Improve the classification accuracy in scenarios of noise or channel impairment.
3. Prompt Formulation Module (Prompt Formulation) constructs a structured prompt consisting of three parts, as shown in Table 1:
•  
Instruction module (I): Specifies that the model should select a classification result from a predefined modulation type list (such as 4ASK, 4PAM, 8ASK, etc.), and outputs only a single valid answer to avoid redundant information.
•  
Example Summary (C): Containing statistical summaries of example signals for each modulation type and corresponding real modulation labels, providing a basis for the model's learning.
•  
Query Part (linearize (x) + S): Composed of the statistical summary of linearization of test signals and the instruction suffix (S) of enhanced modulation category, it guides the model to focus on the characteristics of the signal to be classified.
Table 1: Modulation Classification Suggested Word Structure: Instructor Module, Example Summary, and Query Section
Module
Example content
Instruction module
You are a professional quantitative analyst in the field of wireless communication modulation. Based on your knowledge in wireless communication modulation and the detailed signal statistical information provided below, determine the modulation type. 1. No additional information or background is allowed. 2. Avoid other texts to prevent redundant expressions.
Example Summary
Example 1: {nobs: 1024} min: -7.3, max: 7.5, mean: -0.0, ...},  Answer 1: 4ASK example 2: {nobs: 1024, min: -6.6, max: 7.5, mean: -0.1, ...},  Answer 2: 4PAM Example 3: {nobs: 1024, min: -19.8, max: 15.2, mean: 0.2, ...},  Answer 3: 8ASK (more examples omitted)
Query section
Sample Problem: {nobs: 1024, min: -33.502, max: 32.946, mean: -0.629, ...}
4. Generation and Post-Processing module
Model configuration: After tokenizing the pre-built prompts, input the fine-tuned causal language model and configure generation parameters (such as low temperature, limiting maximum token count) to ensure output determinacy.
•  
Result Extraction: Post-processing the model output to remove irrelevant characters, and extracting the predicted modulation type by matching it with a predefined list of valid modulation types.
•  
Accuracy calculation: Compare the predicted modulation type with the actual label to obtain the classification accuracy.
IV. Conclusion This study proposes a novel context-aware automatic modulation classification method. The method utilizes higher-order statistics and cumulative quantities for estimation. Modulation classification is realized through prompt word classification based on pre-trained large language models. By transforming quantitative signal features into natural language descriptions, This framework does not rely on explicit channel or signal-to-noise ratio inputs. It can effectively capture the intrinsic characteristics of complex signal values under different scenarios.
Experimental results on synthetically generated datasets indicate that incorporating example contexts into prompt words can significantly improve classification accuracy; more extensive models like o3-mini can further enhance performance, especially in noisy environments. 
Overall, this study is a crucial first step towards leveraging large language models for automatic tuning classification. We believe that further developments in prompt word design, model architecture, and domain-specific integration will open up new possibilities for robust and efficient signal processing in wireless communications. 
End

## Images

![(Modulation Identification)Plug-and-Play AMC: Contextualized Open Set Modulation Based on LLMs, with Wang as the context. [Attached Python code]](01.jpg)
![(Modulation Identification)Plug-and-Play AMC: Contextualized Open Set Modulation Based on LLMs, with Wang as the context. [Attached Python code]](02.jpg)
![(Modulation Identification)Plug-and-Play AMC: Contextualized Open Set Modulation Based on LLMs, with Wang as the context. [Attached Python code]](03.jpg)
![(Modulation Identification)Plug-and-Play AMC: Contextualized Open Set Modulation Based on LLMs, with Wang as the context. [Attached Python code]](04.jpg)
![(Modulation Identification)Plug-and-Play AMC: Contextualized Open Set Modulation Based on LLMs, with Wang as the context. [Attached Python code]](05.jpg)
![(Modulation Identification)Plug-and-Play AMC: Contextualized Open Set Modulation Based on LLMs, with Wang as the context. [Attached Python code]](06.jpg)

<!-- PAYMENT_BLOCK -->

## Get this dataset

- Payment link (Stripe): [Buy now](https://buy.stripe.com/3cs8yP7sY87d0vu9AB)
- Payment QR code: ![Payment QR code](pay.png)

<!-- /PAYMENT_BLOCK -->
