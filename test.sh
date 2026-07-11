#!/bin/bash
# python main.py --batch_size=32 --method=clip_lora_eeFrozen  --lr=0.0002 --optim_type=SGD  --dataset=cnnsport
# python main.py --batch_size=32 --method=clip_lora_eeFrozen  --lr=0.0002 --optim_type=SGD  --dataset=assess
# python main.py --batch_size=32 --method=clip_lora_eeFrozen_GMM  --lr=0.0002 --optim_type=SGD  --gpu=0 --dataset=fdmas 
# python main.py --batch_size=32 --method=clip_lora_eeFrozen_DCT2  --lr=0.0002 --optim_type=SGD  --gpu=0 --dataset=fdmas 
# 
# python main.py --batch_size=32 --method=clipmoe_sobel_pass_npr  --lr=0.0002 --optim_type=SGD  --dataset=assess --gpu=2
# python main.py --batch_size=32 --method=clipmoe_sobel_pass_npr  --lr=0.02 --optim_type=SGD  --dataset=fdmas --gpu=3
# python main.py --batch_size=32 --method=clipmoe_sobel_pass_npr  --lr=0.02 --optim_type=AdamW --dataset=fdmas --gpu=1
# python main.py --batch_size=32 --method=clipmoe_sobel_pass_npr  --lr=0.02 --optim_type=Adam --dataset=fdmas --gpu=2
# 
# python inference.py --batch_size=32 --method=clip_lora_eeFrozen  --lr=0.0002 --optim_type=SGD  --dataset=assess --gpu=1
# python inference.py --batch_size=32 --method=clip_lora_eeFrozen_GMM  --lr=0.02 --optim_type=SGD  --dataset=fdmas --gpu=1
# python inference.py --batch_size=32 --method=clip_lora_eeFrozen_DCT2  --lr=0.02 --optim_type=AdamW --dataset=ojha --gpu=0
# # python inference.py --batch_size=32 --method=clip_lora_eeFrozen_GMM  --lr=0.02 --optim_type=Adam --dataset=Chameleon --gpu=1
# python inference.py --batch_size=32 --method=clip_lora_eeFrozen_DCT2  --lr=0.02 --optim_type=Adam --dataset=cnnspot --gpu=0
# python inference.py --batch_size=32 --method=clip_lora_eeFrozen_DCT2  --lr=0.02 --optim_type=Adam --dataset=tan --gpu=0
# 
# python inference.py --batch_size=64 --method=clip_lora_eeFrozen_DCT2  --lr=0.02 --optim_type=SGD  --dataset=fdmas --gpu=1
# python inference.py --batch_size=64 --method=clip_lora_eeFrozen_DCT2  --lr=0.02 --optim_type=AdamW --dataset=ojha --gpu=1
# python inference.py --batch_size=64 --method=clip_lora_eeFrozen_DCT2  --lr=0.02 --optim_type=Adam --dataset=Chameleon --gpu=1
# python inference.py --batch_size=64 --method=clip_lora_eeFrozen_DCT2  --lr=0.02 --optim_type=Adam --dataset=cnnspot --gpu=1
# python inference.py --batch_size=64 --method=clip_lora_eeFrozen_DCT2  --lr=0.02 --optim_type=Adam --dataset=tan --gpu=1
# python inference.py --batch_size=64 --method=clip_lora_eeFrozen_DCT2  --lr=0.02 --optim_type=Adam --dataset=VARsC --gpu=1
# 
# python inference.py --batch_size=64 --method=xception_sobel_pass_npr  --lr=0.02 --optim_type=AdamW --dataset=fdmas --gpu=1
# python inference.py --batch_size=64 --method=xception_sobel_pass_npr  --lr=0.02 --optim_type=AdamW --dataset=ojha --gpu=1
# python inference.py --batch_size=32 --method=xception_sobel_pass_npr  --lr=0.02 --optim_type=Adam --dataset=Chameleon --gpu=1
# python inference.py --batch_size=64 --method=xception_sobel_pass_npr  --lr=0.02 --optim_type=Adam --dataset=cnnspot --gpu=1
# python inference.py --batch_size=64 --method=xception_sobel_pass_npr  --lr=0.02 --optim_type=Adam --dataset=tan --gpu=1
# python inference.py --batch_size=64 --method=xception_sobel_pass_npr  --lr=0.02 --optim_type=Adam --dataset=VARsC --gpu=1
# 
# python inference.py --batch_size=64 --method=clip_lora_eeFrozen_DCT3  --lr=0.02 --optim_type=SGD  --dataset=fdmas --gpu=1
# python inference.py --batch_size=64 --method=clip_lora_eeFrozen_DCT3  --lr=0.02 --optim_type=AdamW --dataset=ojha --gpu=1
# python inference.py --batch_size=64 --method=clip_lora_eeFrozen_DCT3  --lr=0.02 --optim_type=Adam --dataset=Chameleon --gpu=1
# python inference.py --batch_size=64 --method=clip_lora_eeFrozen_DCT3  --lr=0.02 --optim_type=Adam --dataset=cnnspot --gpu=1
# python inference.py --batch_size=64 --method=clip_lora_eeFrozen_DCT3  --lr=0.02 --optim_type=Adam --dataset=tan --gpu=1
# python inference.py --batch_size=64 --method=clip_lora_eeFrozen_DCT3  --lr=0.02 --optim_type=Adam --dataset=VARsC --gpu=1
# 
# python inference.py --batch_size=64 --method=Xception_DCT  --lr=0.02 --optim_type=SGD  --dataset=fdmas --gpu=1
# python visualization.py --batch_size=64 --method=clip_lora_eeFrozen_DCT4  --lr=0.02 --optim_type=SGD  --dataset=fdmas --gpu=1
# python visualcam.py --batch_size=64 --method=clip_lora_eeFrozen_DCT4  --lr=0.02 --optim_type=SGD  --dataset=fdmas --gpu=1
# python visualization.py --batch_size=64 --method=xception_sobel_pass_npr  --lr=0.02 --optim_type=SGD  --dataset=fdmas --gpu=1
# python visualization.py --batch_size=64 --method=Xception_DCT  --lr=0.02 --optim_type=SGD  --dataset=fdmas --gpu=1
# python inference.py --batch_size=64 --method=clip_lora_eeFrozen_DCT4  --lr=0.02 --optim_type=SGD  --dataset=fdmas --gpu=1
# python inference.py --batch_size=64 --method=clip_lora_eeFrozen_DCT4  --lr=0.02 --optim_type=AdamW --dataset=ojha --gpu=1
# python inference.py --batch_size=64 --method=clip_lora_eeFrozen_DCT4  --lr=0.02 --optim_type=Adam --dataset=Chameleon --gpu=1
# python inference.py --batch_size=64 --method=clip_lora_eeFrozen_DCT4  --lr=0.02 --optim_type=Adam --dataset=cnnspot --gpu=1
python inference.py --batch_size=64 --method=clip_lora_eeFrozen_DCT4  --lr=0.02 --optim_type=Adam --dataset=tan --gpu=1
# python inference.py --batch_size=64 --method=clip_lora_eeFrozen_DCT4  --lr=0.02 --optim_type=Adam --dataset=VARsC --gpu=1
# python inference.py --batch_size=256 --method=clip_lora_eeFrozen_DCT4  --lr=0.02 --optim_type=Adam --dataset=ARForensics --gpu=1
# python inference.py --batch_size=64 --method=clip_lora_eeFrozen_DCT4  --lr=0.02 --optim_type=Adam --dataset=WildRF --gpu=1
# 
# python inference_dis.py --batch_size=64 --method=xception_sobel_pass_npr  --lr=0.02 --optim_type=SGD  --dataset=fdmas --gpu=1
# python inference_dis.py --batch_size=64 --method=xception_sobel_pass_npr  --lr=0.02 --optim_type=AdamW --dataset=ojha --gpu=1
# python inference_dis.py --batch_size=64 --method=xception_sobel_pass_npr  --lr=0.02 --optim_type=Adam --dataset=Chameleon --gpu=1
# python inference_dis.py --batch_size=64 --method=xception_sobel_pass_npr  --lr=0.02 --optim_type=Adam --dataset=cnnspot --gpu=1
# python inference_dis.py --batch_size=64 --method=xception_sobel_pass_npr  --lr=0.02 --optim_type=Adam --dataset=tan --gpu=1
# python inference_dis.py --batch_size=64 --method=xception_sobel_pass_npr  --lr=0.02 --optim_type=Adam --dataset=VARsC --gpu=1