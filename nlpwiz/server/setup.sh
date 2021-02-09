

sudo apt-get install rabbitmq-server

sudo rabbitmq-plugins enable rabbitmq_management
#Once you've enabled the console, it can be accessed using your favourite web browser by visiting: http://[your droplet's IP]:15672/


#To have RabbitMQ start as a daemon by default, run the following:
chkconfig rabbitmq-server on

sudo service rabbitmq-server start