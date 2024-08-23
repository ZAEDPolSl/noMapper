<center>  

```
                  888b     d888                                             
                  8888b   d8888                                             
                  88888b.d88888                                             
88888b.   .d88b.  888Y88888P888  8888b.  88888b.  88888b.   .d88b.  888d888 
888 "88b d88""88b 888 Y888P 888     "88b 888 "88b 888 "88b d8P  Y8b 888P"   
888  888 888  888 888  Y8P  888 .d888888 888  888 888  888 88888888 888     
888  888 Y88..88P 888   "   888 888  888 888 d88P 888 d88P Y8b.     888     
888  888  "Y88P"  888       888 "Y888888 88888P"  88888P"   "Y8888  888     
                                         888      888                       
                                         888      888                       
                                         888      888                       
```

</center>

## Description
Implementation of a selector based on NLP techniques that searches for a specific sequence among DNA/RNA long-reads. The software does not use alignment algorithms (such as Needleman-Wunsch or Smith-Waterman) in its work and is a *'noMapping mapping'* solution.

## ▸ Prepare noMapper ◂
### Getting Started
1. Go to the relevant directory
    ```bash
    cd docker/nomapper-maker/
    ```
2. Insert in the `vol/` directory:
    - `input.fastq.gz` - sequences for training the model
    - `ref.fna` - a reference, i.e. the sequence you want to search for ([an example](https://www.ncbi.nlm.nih.gov/nuccore/NC_000017.11?report=fasta&from=74862497&to=74872994&strand=true))
3. Download latest stable version
    ```bash
    docker pull drdext3r/nomapper-maker
    ```
4. Run the docker container
    ```bash
    docker run -it -v $(pwd)/vol:/vol --name nomapper-maker drdext3r/nomapper-maker:latest
    ```
5. Train the model  
    with default configuration
    ```bash
    do-all
    ```
    or custom configuration   
    ```bash
    preprocess
    python3 train.py --help     # show configuration
    python3 train.py --kmer_size=4 
    chmod 777 /vol/model.h5
    ```    
6. Clean out unnecessary files (optional)
    ```bash
    clean-all
    ```
7. Exit the docker container
    ```bash
    exit
    ```
### Outcome
- `model.h5` - the trained model
- `cv.pkl` - the encoder


## ▸ Use noMapper ◂
### Getting Started
1. Go to the relevant directory
    ```bash
    cd docker/nomapper/
    ```
2. Insert in the `vol/` directory
    - `model.h5` - the trained model
    - `cv.pkl` - the encoder
3. Set the configuration file `vol/config.ini`
4. Download latest stable version
    ```bash
    docker pull drdext3r/nomapper
    ```
5. Run the docker container
    ```bash
    docker run -it -v $(pwd)/vol:/vol -p 8000:8000 --name nomapper drdext3r/nomapper:latest
    ```
6. Predict in a new window
    ```bash
    curl -X POST "http://127.0.0.1:8000/predict/" -H "accept: application/json" -H "Content-Type: application/json" -d '{"seq": "<long-read>"}'
    ```
    Outputs:
    ```json
    {"result":"found"}
    ```
    ```json
    {"result":"not found"}
    ```
7. Exit the docker container
    ```bash
    exit
    ```
