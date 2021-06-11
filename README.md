# Learning Loss for Active Learning
 Reproducing experimental results of LL4AL [Yoo et al. 2019 CVPR]

# Reproduced Results
 ![Results](./results.PNG)

# Requirements
 torch >= 1.1.0

 numpy >= 1.16.2

 tqdm >= 4.31.1

 visdom >= 0.1.8.8

# To Activate Visdom Server
  visdom -port 9000

  or 

  python -m visdom.server -port 9000

# Contact
 ciy405x@kaist.ac.kr
 
 # Jupyter
 jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com' --port=8888 --NotebookApp.port_retries=0