

sudo service rabbitmq-server start

export PYTHONPATH=$HOME/src/nlpwiz:$PYTHONPATH

nameko run model_service