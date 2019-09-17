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