"""初始版本测试阶段

单图：
python run_local_infer.py images\s3.jpg --prompt-file sample_prompt.txt

python run_local_infer.py images\s3.jpg --prompt-file sample_prompt.txt --base-url http://你的IP:8002

文件夹：
python run_local_infer.py images --prompt-file sample_prompt.txt --recursive

"""



"""能够进行流式传输测试

python controller_generate.py images\s3.jpg --prompt-file controller_prompt.txt --do-sample --candidates-per-round 3 --max-sentences 3 --max-new-tokens 96 --temperature 0.8 --top-p 0.9 --top-k 50


"""