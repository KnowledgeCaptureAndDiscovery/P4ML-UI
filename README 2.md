# DSBox TA2 System

## 1. Make sure your git credentials are set
    git config --global credential.https://gitlab.datadrivendiscovery.org.username [gitlab_user_id]
    git config --global credential.https://github.com.username [github_user_id]

## 2. Pull recursive
    git clone --recursive https://github.com/usc-isi-i2/dsbox-ta2-system

## 3. Creating the ISI Docker Image
Readme borrowed from: https://github.com/usc-isi-i2/dsbox-ta2/blob/docker-ta2/README.md

This section details the distilled commands that need to be run to create the docker image. The documentation mentioned
above is very good but contains a lot of options and variables that need to be navigated in order to come up with the
correct credentials, urls and commands.

1. In the same directory as this readme file run the following:
    
    ```
    docker login registry.datadrivendiscovery.org 
    docker build -f docker/python3/Dockerfile -t registry.datadrivendiscovery.org/ta2/isi_ta2:latest .
    ```
    
2. Get the id of the image that was just created for use in the next step:

    ```
    > docker images
    REPOSITORY                   TAG                 IMAGE ID            CREATED             SIZE
    ../ta2/isi_ta2               latest              8787d9aa6999        9 hours ago         2.43GB
    ``` 
    
    
3. To test the docker image before pushing to the nist docker registry, run the following command. Notice the paths to 
   config.json - the local copy of the file will need the absolute path. This also goes for the sample data set that
   you want to run (in the case below, baseball)

    ``` 
    > docker run -it --entrypoint /bin/bash \
    -v /tmp/conf/search-185.conf:/tmp/config.json \
    -v /tmp/results:/tmp/results \
    -v /Users/varun/git/dsbox/data:/tmp/data registry.datadrivendiscovery.org/ta2/isi_ta2:latest \
    -c 'ta2_search /tmp/config.json'
    ```

4. To evaluate the test executable, check in the results folder for an executable to run:

    ``` 
    > docker run -it --entrypoint /bin/bash \
    -v /tmp/conf/test-185.conf:/tmp/config.json \
    -v /tmp/results:/tmp/results \
    -v /tmp/data:/tmp/data registry.datadrivendiscovery.org/ta2/isi_ta2:latest \
    -c '/tmp/results/o_185/executables/1a8331f7-8a08-40bc-930c-c83ce684d7f9 /tmp/config.json'
    ```

5. To run the TA2 server, run docker like so:

    ``` 
    > docker run -v /Users/Varun/git/dsbox/data:/tmp/data \
    -v /tmp/dsbox-ta2:/tmp/dsbox-ta2 \
    -p 45042:45042 \
    -d registry.datadrivendiscovery.org/ta2/isi_ta2:latest
    ```

6. If the docker run completes with pipelines created then you are ready to upload the image to the NIST Docker Registry.
   If you do not have an account for this let Daragh Hartnett (daragh.hartnett@sri.com) know and he will get you the 
   credentials you need. Run the following commands:

   ```  
   > docker login registry.datadrivendiscovery.org
   > docker push registry.datadrivendiscovery.org/ta2/isi_ta2:latest 
   ```
   
7. Run the Docker CI pipeline by opening the following in a browser: 

      https://gitlab.datadrivendiscovery.org/TA2/ISI_ta2
    
   - Click on 'Pipelines' in the top menu and then 'Pipelines' in the sub menu. 
   - Click on Run Pipeline
   - Click on Create Pipeline
   - Click on the Play button in the setup node.
   - In the logs, you should be able to see the latest image id (63e81a8c1fed for example) embedded in the long 
   identifier for the latest docker image being used (Using docker image registry.datadrivendiscovery.org/ta2/isi_ta2:
   latest ID=sha256:**63e81a8c1fed**c2beabb2e8ee3b2cb84b4ed96dc97ebc071d0a2217d984716708...)
   
