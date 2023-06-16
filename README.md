## langchain-sql-databricks

**TLDR;** this repo contains some starter examples for working with Langchain and LLM Instruction models (e.g. MPT, from MosaicML) to query Databricks SQL.

### Getting started

* Create a Databricks single node cluster with a GPU (e.g. `Standard_NC8as_T4_v3` on Azure or `g4dn.xlarge` on AWS). Select at runtime version 13.0 ML with GPU support (minimum).

* The `config` folder contains an init script. Configure your cluster to use this init script by pointing to it either through the Cluster UI or Terraform.

* Install the following packages into the cluster:

    * accelerate
    * databricks-sql-connector
    * flash-attn==1.0.5
    * langchain
    * ninja
    * transformers==4.29.2
    * triton

You should be good to go.
