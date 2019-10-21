# Cheat Sheet

The official tutorial is very dev-ops heavy most of it is not so important for datascience
extracts from [official tutorial](https://docs.docker.com/get-started/)

This [R docker tutorial](https://ropenscilabs.github.io/r-docker-tutorial/) is much better

## List Docker CLI commands
```
docker
docker container --help
```

## Display Docker version and info
```
docker --version
docker version
docker info
```

## Execute Docker image
```
docker run hello-world
docker run --rm -it sample_image /bin/bash # run in interactive mode and quit container when exiting attach shell
docker run --rm -it -v local_path:container_path sample_image /bin/bash  # attach volume
```

## List Docker images
```
docker image ls
```

## List Docker containers (running, all, all in quiet mode)
```
docker container ls
docker container ls --all
docker container ls -aq
```

## Managin Images and running containers
```
docker build -t friendlyhello .  # Create image using this directory's Dockerfile
docker run -p 4000:80 friendlyhello  # Run "friendlyhello" mapping port 4000 to 80
docker run -d -p 4000:80 friendlyhello         # Same thing, but in detached mode
docker container ls                                # List all running containers
docker container ls -a             # List all containers, even those not running
docker container stop <hash>           # Gracefully stop the specified container
docker container kill <hash>         # Force shutdown of the specified container
docker container rm <hash>        # Remove specified container from this machine
docker container rm $(docker container ls -a -q)         # Remove all containers
docker image ls -a                             # List all images on this machine
docker image rm <image id>            # Remove specified image from this machine
docker image rm $(docker image ls -a -q)   # Remove all images from this machine
docker login             # Log in this CLI session using your Docker credentials
docker tag <image> username/repository:tag  # Tag <image> for upload to registry
docker push username/repository:tag            # Upload tagged image to registry
docker run username/repository:tag                   # Run image from a registry
```

# Dockerfile

## Install linux modules
- `&&` chains two commands
- `\` continue command on next line
- `-y` passes yes to user input
```
RUN apt-get update && \
    apt-get -y install build-essential && \
    apt-get -y install openssh-client
```

## Install into conda environment
```
RUN conda create --name my_env python=3.6
RUN conda install -n my_env pyarrow=0.11.1
ENV PATH /opt/conda/envs/my_env/bin:$PATH
RUN /bin/bash -c "source activate idwimpala"
RUN pip install hdfs==2.1.0
```
## RUN, CMD, ENTRYPOINT

- `RUN` executes line, execution can be stored in image
- `ENTRYPOINT` executes when container starts up
- `CMD` default command when container starts up, can be overwritten

[blogpost](https://goinbigdata.com/docker-run-vs-cmd-vs-entrypoint/)

## Run RStudio

- `-p 8787:8787` defines port
- `-e PASSWORD='123'` sets environmen variable PASSWORD before running container
- `-d` detached mode, runs in background
- `/init` initiates RStudio
```
docker run --rm -it -p 8787:8787 -v local_path:/home/rstudio/container_path -e PASSWORD='123' -d rocker/verse /init
```

# Deploy Services

specifications such as remote image location, ressource allocation and number of instances
can be specified in `docker-compose.yml`

```
docker swarm init         # needs to be executed in order to deploy the services
docker stack ls                                            # List stacks or apps
docker stack deploy -c <composefile> <appname>  # Run the specified Compose file
docker service ls                 # List running services associated with an app
docker service ps <service>                  # List tasks associated with an app
docker inspect <task or container>                   # Inspect task or container
docker container ls -q                                      # List container IDs
docker stack rm <appname>                             # Tear down an application
docker swarm leave --force      # Take down a single node swarm from the manager
```
