# SwiftLLM

*This project is still under development. Some features may not be implemented yet, and documentation may be incomplete.*

A tiny yet powerful LLM inference system tailored for researching purpose.

## Why SwiftLLM

There are so many open source frameworks for LLM serving, including [HuggingFace Transformers](https://github.com/huggingface/transformers), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), and [DeepSpeed-MII](https://github.com/microsoft/DeepSpeed-MII). Why SwiftLLM?

The reason is that, those frameworks are tailored for **production**, instead of **researching**. They are equipped with numerous features, such as 100+ model supports, various hardward supports, LoRA, quantization, multimodal, prefix caching, beam search, and so on. While being an all-in-one solution for production, their codebase is too big and complex to understand and modify (for example, vLLM has 100k+ lines of code), making it hard to use them for researching purpose. Also, their historical burden is also a problem.

SwiftLLM is designed to be a tiny yet powerful LLM inference system tailored for **researching purpose**. "Tiny" means that it only keeps features that are essential for researching, "powerful" means that it has no compromise on performance, and finally "swift" means that it is easy to understand and modify. While supporting basic features (see the list below) and being able to achieve equivalent performance to vLLM, the codebase of SwiftLLM is less than **2k** lines of code, written in Python and OpenAI Triton (a DSL for writing CUDA kernels), making it easy to read, modify, debug, test, extend, and can be easily integrated with your novel and brilliant research ideas.

## Feature List

Currently, SwiftLLM supports the following features:

- Iterational Scheduling and Selective Batching (proposed in [Orca](https://www.usenix.org/conference/osdi22/presentation/yu))
- PagedAttenton (proposed in [vLLM](https://github.com/vllm-project/vllm), [paper](https://arxiv.org/abs/2309.06180))
- LLaMA / LLaMA2 / LLaMA3 models ([link](https://llama.meta.com/))
- Piggybacking prefill and decoding (proposed in [SARATHI](https://arxiv.org/abs/2308.16369))
- Flash attention (proposed in [FlashAttention](https://arxiv.org/abs/2205.14135) and [FlashAttention-2](https://arxiv.org/abs/2307.08691))
- Paged attention v2 (also called Flash-Decoding, proposed [here](https://crfm.stanford.edu/2023/10/12/flashdecoding.html))

And we plan to add support for the following features in the future:

- Tensor parallelism and pipeline parallelism