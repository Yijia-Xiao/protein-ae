python run.py  --loss spear --optim adamw
python run.py  --loss mse --optim adamw
python run.py  --loss spear --optim sgd
python run.py  --loss mse --optim sgd
bash s1.sh
bash ae1.sh
bash ae2.sh
(python main.py) |& tee -a log
