# Intel&reg; Gaudi&reg; vLLM Tutorials

The tutorials provide step-by-step instructions for running [vLLM](https://github.com/HabanaAI/vllm-fork/blob/habana_main/README.md) inference server on the Intel Gaudi AI Processor, from beginner level to advanced users.  These tutorials should be run with a full Intel Gaudi Node of 8 cards. 

## IMPORTANT: To run most of these Jupyter Notebooks you will need to follow these steps:
1. Get access to an Intel Gaudi 2 Accelerator card or node. See the [Get Access](https://developer.habana.ai/get-access/) page on the Developer Website.  Be sure to use port forwarding `ssh -L 8888:localhost:8888 -L 7860:localhost:7860 -L 6006:localhost:6006 ... user@ipaddress` to be able to access the notebook, run the Gradio interface, and use Tensorboard. Some of the tutorials use all of these features.
2. 
    a) If interested in running the Getting Started with vLLM tutorial, refer Steps 4 and 5 to install jupyterlab and run the tutorial notebook directly (Skip the rest of these instructions).

    b) (**NOTE**: Run this and subsequent steps only if not running the Getting Started with vLLM tutorial above) Run the Intel Gaudi PyTorch Docker image. Refer to the Docker section of the [Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html#bare-metal-fresh-os-single-click) for more information.  Running the docker image will allow you access to the entire software stack without having to worry about detailed Software installation Steps.
```
docker run -itd --name Gaudi_Docker --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host vault.habana.ai/gaudi-docker/1.21.0/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:1.21.0-555
docker exec -it Gaudi_Docker bash
```
3. Clone this tutorial in your $HOME directory:  `cd ~ && git clone https://www.github.com/habanaAI/vllm-tutorials`
4. Install Jupyterlab: `python3 -m pip install jupyterlab`
5. Run the Jupyterlab Server, using the same port mapping as the ssh command:  `python3 -m jupyterlab_server --IdentityProvider.token='' --ServerApp.password='' --allow-root --port 8888 --ServerApp.root_dir=$HOME & ` and take the local URL and run that in your browser

The tutorials will cover the following domains and tasks:

### Advanced
- [Deploying on vLLM](https://github.com/HabanaAI/Gaudi-tutorials/tree/main/PyTorch/vLLM_Tutorials/Deploying_vLLM)

### Intermediate
- [Benchmarking on vLLM](http://localhost:9010/lab/tree/Gaudi-tutorials/PyTorch/vLLM_Tutorials/Benchmarking_on_vLLM/vLLM_Benchmark_Serving.ipynb)

### Getting Started
- [Getting Started with vLLM](https://github.com/HabanaAI/Gaudi-tutorials/blob/main/PyTorch/vLLM_Tutorials/Getting_Started_with_vLLM/Getting_Started_with_vLLM.ipynb)
- [Understanding vLLM on Gaudi](https://github.com/HabanaAI/Gaudi-tutorials/blob/main/PyTorch/vLLM_Tutorials/Understanding_vLLM_on_Gaudi/Understanding_vLLM_on_Gaudi.ipynb)

## DISCLAIMER
Intel is committed to the respect of human rights and avoiding complicity in human rights abuses, a policy reflected in the Intel Global Human Rights Principles. Accordingly, by accessing the Intel material on this platform you agree that you will not use the material in a product or application that causes or contributes to a violation of an internationally recognized human right.

### License
Intel® Gaudi® vLLM Tutorials is licensed under Apache License Version 2.0.

### Datasets and Models
To the extent that any data, datasets, or models are referenced by Intel or accessed using tools or code on this site such data, datasets and models are provided by the third party indicated as the source of such content. Intel does not create the data, datasets, or models, provide a license to any third-party data, datasets, or models referenced, and does not warrant their accuracy or quality. By accessing such data, dataset(s) or model(s) you agree to the terms associated with that content and that your use complies with the applicable license.

Intel expressly disclaims the accuracy, adequacy, or completeness of any data, datasets or models, and is not liable for any errors, omissions, or defects in such content, or for any reliance thereon. Intel also expressly disclaims any warranty of non-infringement with respect to such data, dataset(s), or model(s). Intel is not liable for any liability or damages relating to your use of such data, datasets, or models.
