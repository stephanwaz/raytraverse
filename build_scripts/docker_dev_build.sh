
python setup.py sdist
docker build -f build_scripts/DockerfileDev --tag raytedit .
docker run -d -it --name rayt1 --mount type=bind,source="$(pwd)"/src,target=/hostsrc raytedit
docker exec -it rayt1 /bin/bash
echo usage: 
echo docker exec -it rayt1 /bin/bash
echo copy changes into /hostsrc
echo docker stop -t 0 rayt1
echo docker rm rayt1

