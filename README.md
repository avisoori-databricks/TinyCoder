# TinyCoder
Code to train TinyCoder-1B-Instruct

Goal: 

Craft a high quality instruction following dataset for code synthesis and instruction following (~90k in total): Used a high quality subset of Dolly-15k augmented to stylistically resemble responses from an intelligent assistant (as shown in LIMA: Less Is More for Alignment ), WikiSQL, Spider, subset of Rosetta Code, Strarcoder Self-Instruct (curated set) and Grade School Math Instructions.

Instruction finetune CodeGen2 -1B with QLoRA on a single V100 (optimized for better model performance). 

Build a simple app by leveraging TinyCoder deployed on GPU serving. Solara is used for developing the interface. (Solara documentation )

 

Result: TinyCoder-1B-Instruct, finetuned for 10 epochs on a singe V100 on the above mix of data (~90k text-response pairs styled in the Alpaca format) performs reasonably well for generating small python functions. Although outside the scope of this hackathon project, it would be interesting to scale the model size to 3B and see how the model performs. 

Tips for getting QLoRA to actually be performant:

Tips for effective training with QLoRA (primarily avoiding overfitting and ensuring generalization when using LoRA):

target_modules should include all linear layers instead of just the attention blocks. This increases training time substantially, but still much faster than full finetuning (highest impact)

Opt for a higher rank for the low rank matrices e.g. r=16 vs r=8

Upcast the layer norms to float 32 for more stable training

Use a memory efficient and stable optimizer such as AdamW

Inputs follow the Alapaca format:


Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
 
