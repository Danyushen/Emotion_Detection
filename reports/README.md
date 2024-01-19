---
layout: default
nav_exclude: true
---


# Exam template for 02476 Machine Learning Operations


This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:


```--- question 1 fill here ---```


where you instead should add your answers. Any other changes may have unwanted consequences when your report is auto
generated in the end of the course. For questions where you are asked to include images, start by adding the image to
the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:


```markdown
![my_image](figures/<image>.<extension>)
```


In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:


Running:


```bash
python report.py html
```


will generate an `.html` page of your report. After deadline for answering this template, we will autoscrape
everything in this `reports` folder and then use this utility to generate an `.html` page that will be your serve
as your final handin.


Running


```bash
python report.py check
```


will check your answers in this template against the constrains listed for each question e.g. is your answer too
short, too long, have you included an image when asked to.


For both functions to work it is important that you do not rename anything. The script have two dependencies that can
be installed with `pip install click markdown`.


## Overall project checklist


The checklist is *exhaustic* which means that it includes everything that you could possible do on the project in
relation the curricilum in this course. Therefore, we do not expect at all that you have checked of all boxes at the
end of the project.


### Week 1


* [x] Create a git repository
* [x] Make sure that all team members have write access to the github repository
* [x] Create a dedicated environment for you project to keep track of your packages
* [x] Create the initial file structure using cookiecutter
* [x] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
* [x] Add a model file and a training script and get that running
* [x] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
* [x] Remember to comply with good coding practices (`pep8`) while doing the project
* [x] Do a bit of code typing and remember to document essential parts of your code
* [x] Setup version control for your data or part of your data
* [x] Construct one or multiple docker files for your code
* [x] Build the docker files locally and make sure they work as intended
* [x] Write one or multiple configurations files for your experiments
* [x] Used Hydra to load the configurations and manage your hyperparameters
* [ ] When you have something that works somewhat, remember at some point to to some profiling and see if
      you can optimize your code
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code. Additionally,
      consider running a hyperparameter optimization sweep.
* [x] Use Pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code


### Week 2


* [x] Write unit tests related to the data part of your code
* [x] Write unit tests related to model construction and or model training
* [x] Calculate the coverage.
* [x] Get some continuous integration running on the github repository
* [x] Create a data storage in GCP Bucket for you data and preferable link this with your data version control setup
* [x] Create a trigger workflow for automatically building your docker images
* [x] Get your model training in GCP using either the Engine or Vertex AI
* [x] Create a FastAPI application that can do inference using your model
* [ ] If applicable, consider deploying the model locally using torchserve
* [ ] Deploy your model in GCP using either Functions or Run as the backend


### Week 3


* [ ] Check how robust your model is towards data drifting
* [ ] Setup monitoring for the system telemetry of your deployed model
* [ ] Setup monitoring for the performance of your deployed model
* [ ] If applicable, play around with distributed data loading
* [ ] If applicable, play around with distributed model training
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed


### Additional


* [x] Revisit your initial project description. Did the project turn out as you wanted?
* [x] Make sure all group members have a understanding about all parts of the project
* [x] Uploaded all your code to github


## Group information


### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:


15


### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer: 
> s193396, s204165, s103629, s240485, s204127


### Question 3
> **What framework did you choose to work with and did it help you complete the project?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer: 
> The project leverages the TIMM, third-party framework, chosen for its extensive range of models, including pretrained ones. In our implementation, TIMM plays a central role in both constructing the model and incorporating pretrained features, forming the backbone of our architecture. 
> Additionally, PyTorch Lightning proves instrumental in simplifying the model-building and training processes, eliminating the need for standard PyTorch boilerplate code. It serves a dual purpose by defining the architecture during model construction and facilitating training settings, such as specifying the number of GPUs, choosing the accelerator (CPU, GPU, or TPU), determining the maximum number of epochs for training, and initializing the training. The integration of Wandblogger enhances the project's logging capabilities.
>
> Hydra is used to streamline the handling of parameters for training, prediction and web API. It establishes a connection with a config file containing relevant hyperparameters, offering a centralized and efficient means of controlling hyperparameters. For data preparation, we use Albumentations, an image transformations library. Within a custom transformer class, we've established an easily expandable pipeline. Currently, it resizes images to a constant resolution, but it can be easily expanded with functions provided by Albumentations. This approach ensures flexibility in adapting the transformation process as needed.


## Coding environment


> In the following section we are interested in learning more about you local development environment.


### Question 4


> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:
> We streamlined dependency management in our project using a requirements.txt file. Throughout the project, the list of dependencies was dynamically updated whenever a new package was installed. This ensured that all team members could easily synchronize their local environments with the latest additions.
>
>
> To capture the complete snapshot of our development environment, we generated a comprehensive requirements.txt (and others for development purposes) file at the project's conclusion using the command: pip freeze > requirements.txt. To replicate our exact environment through Conda, one would execute the command: conda create --name <env_name> --file requirements.txt. On the other hand with python venv, one would execute command: python -m venv <env_name> and then command: python -m pip install -r requirements.txt.
>
>
>Furthermore, we organized custom developer and testing modules using requirements_dev.txt and requirements_test.txt, respectively. This modular approach allows team members to manage specific environments tailored to development or testing needs.
>
>
>In summary, obtaining an exact copy of our project's environment involves creating a Conda (or standard python environment) with the specified requirements.txt file, providing a seamless and reproducible setup for new team members.




### Question 5


> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. Did you fill out every folder or only a subset?**
>
> Answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
> Answer:
> The project structure closely aligns with the cookiecutter template, while systematically excluding unneeded folders, these folders are deliberately removed, thereby avoiding having redundant and empty directories. This customization of the project structures ensures that it has precisely those folders needed. This means that specific folders namely ‘models’, ‘notebooks’, and ‘src/visualizations’, have been intentionally removed from the project structure. 
> Instead of the excluded folders, the Hydra framework has created a new folder named ‘outputs’. This folder, organized by date and time for each run, serves as a repository for relevant information for each run. For instance, each subfolder within ‘outputs’ contains relevant information like the config-file used for that specific run. 
> To prevent GIT to be drowned in numerous folders that may not be of interest to all, the ‘outputs’ folder has been added to the .gitignore file. This ensures that information specific to individual runs is saved locally, thereby preventing GIT to be burdened with unneeded data. *


### Question 6


> **Did you implement any rules for code quality and format? Additionally, explain with your own words why these**
> **concepts matters in larger projects.**
>
> Answer length: 50-100 words.
>
> Answer:
> In our project, we followed PEP8 guidelines for code quality, using the ruff code formatter to ensure compliance. Commands like ruff check and ruff format were applied to specific code sections. Simplifying the process, ruff check --fix && ruff format efficiently maintained PEP8 standards. 
In larger projects, adhering to coding practices becomes crucial for consistency across developers, enhancing readability and collaboration. Utilizing a code formatter like ruff automates compliance, promoting a coherent and maintainable codebase as the project scales, minimizing potential issues stemming from inconsistent coding styles.


## Version control


> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.


### Question 7


> **How many tests did you implement and what are they testing in your code?**
>
> Answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:
> In total we implemented six tests, each focusing on different critical aspects of our application to ensure its robustness. We began by ensuring the integrity of our data loading processed through [test_data_loading](../tests/test_data.py), which checked the correctness of our dataset sizes in terms of the dataset split. The  [test_model_input_output_shape](../tests/test_model.py) then verified that our model correctly handled input and output shapes, a critical step for the model's functionality and to gain insight in the model structure. To test the fundamental operational extent of our model, we employed [test_model_prediction](../tests/test_predict_model.py), which assessed the model's ability to make predictions on dummy data. The robustness of the model was further tested in [test_model_with_varied_input_sizes](../tests/test_predict_model.py), ensuring it could handle input of different batch sizes. At last, [test_model_with_real_data](../tests/test_predict_model.py), used actual model files and real data to perform predictions, crucial for real-world applications and to test its ability to make predictions on true data. Finally, [test_train_initialization ](../tests/test_training.py) inspected the initial setup process of our training pipeline, including crucial components like WandB, model instantiation, and data loaders, ensuring a smooth and error-free start to the training process.




### Question 8


> **What is the total code coverage (in percentage) of your code? If you code had an code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:
> The total code coverage of our code was 79%, which included critical components like the model definition, training, prediction, and tests. While we have achieved thorough coverage in key ares, including 97% in [train_model.py](../src/train_model.py) and 100% in [test_training.py](../tests/test_training.py), there are still portions of the code, particularly in [model.py](../src/models/model.py) achieving a diminishing 46%, that lacked full test coverage.
>
> However, it is important to note that code coverage sometimes includes files that aren't necessarily relevant for coverage analysis of tests. For instance, the entire [model.py](../src/models/model.py) might contain parts of the code which weren’t appropriate for the testing scripts. As mentioned in previous Question 7, we were focusing on testing model input and output shape, thus meaning that parts of the model structure would be redundant in covering.
>
> Even if we were to achieve close to 100% code coverage, it wouldn't necessarily imply that the code is error-free. Superior code coverage only indicates that significant portions of the code is utilized by tests, but it doesn't guarantee the detection of bugs, issues, and errors. While striving for high code coverage is beneficial for increasing confidence in the code's reliability, it should be complemented with code reviews for instance.




### Question 9


> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:
> Our project workflow was designed for flexibility, accommodating diverse team preferences through the utilization of branches and forks. Team members had the option to work in individual branches within the project repository, alongside the main branch. Alternatively, some opted for forks of the project repository. To integrate code into the main branch, we implemented pull requests, subject to thorough reviews by other team members.
>
> Ensuring synchronization and continuity, each team member continuously pulled from the main branch of the project repository to update their individual branches or forked repositories. This practice aimed to prevent deprecated code and resolve potential merge conflicts promptly. The combination of individual branches, forks, and pull requests facilitated a collaborative and organized development process, allowing team members to work in parallel while maintaining a structured and cohesive codebase. This approach not only accommodated varying working styles within the team but also contributed to a smooth and well-coordinated project development lifecycle.*




### Question 10


> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:
> Data Version Control (DVC) is used in the project as much as an integrated part like git is used, making version control clear and reliable. The project makes use of this in combination with first google drive, but later scaled up to use bucket through google cloud instead. For setting up the project to be able to make use of DVC, a dedicated folder and a config-file is needed. For the folder it needs to be named like the following: ‘.dvc’, where the config-file is located, which contains information about type of storage, in this case remote-storage, but more importantly where the storage is located. Additionally, a connection between git and google needs to be established using SECRETS, so they have the possibility to exchange data. Having the setup in order and ready to use, the user is then able to run ‘dvc push’ ‘dvc add’ and ‘dvc pull’ commands to either push new data to the storage or to update the data locally.
> In the initial step of the project, the DVC ensured that each team member works with the same data. Later in the project, when the project is moved to the cloud, the DVC ensures that the docker runs on the correct data while also having the trained model version controlled. Using DVC made it possible for the project to work on the correct data throughout the development, while later in the process also include the trained model.




### Question 11


> **Discuss you continues integration setup. What kind of CI are you running (unittesting, linting, etc.)? Do you test**
> **multiple operating systems, python version etc. Do you make use of caching? Feel free to insert a link to one of**
> **your github actions workflow.**
>
> Answer length: 200-300 words.
>
> Example:
> *We have organized our CI into 3 separate files: one for doing ..., one for running ... testing and one for running*
> *... . In particular for our ..., we used ... .An example of a triggered workflow can be seen here: <weblink>*
>
> Answer:
> Our project employs a comprehensive CI setup using GitHub Actions, which ensures the stability and reliability of our codebase through two workflows [Submit Build](../.github/workflows/submit_build.yaml)and [Run Tests](../.github/workflows/tests.yaml). The [Submit Build](../.github/workflows/submit_build.yaml) workflow is a crucial part of our CI pipeline, triggered on both push and pull requests to the remote_storage and main branches. This workflow is designed to integrate seamlessly with Google Cloud, starting with authentication and followed by the submission of a build to the Google Cloud build environment. This automation is essential for maintaining an efficient and consistent docker build process, ensuring that our docker images are always up-to-date with the latest code changes. It simplifies the deployment process and minimizes the potential for human error during the build process.
> The [Run Tests](../.github/workflows/tests.yaml) workflow is focused on unit testing with pytest to verify the integrity of our code. It manages data dependencies by pulling necessary data from Google Cloud Storage using DVC, as well as directly from a specific bucket when required. Following the data setup, the workflow runs a series of tests using pytest. These tests cover various aspects of our application to ensure that new code submissions don’t introduce issues and break existing functionalities. This workflow is a critical aspect, helping to catch bugs and issues early in the development cycle.
> Overall, the CI setup is integral to maintaining the quality of our project. They ensure that every change is thoroughly tested and that the build process remains consistent and automated. The setup enhances robustness in our application, and employs additional code quality checks.


## Running code and tracking experiments


> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.


### Question 12


> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: python my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:
> We used Hydra to manage and load our hyperparameters from a configuration file config.yaml. The hyperparameters specified in the config file are automatically loaded into the train_model.py and predict_model.py scripts, hence by simply running: python train_model.py. Any hyperparameter can be changed for a specific run from the command-line in the following way: python train_model.py hyperparameters.num_epochs=10. 
> Then we even expanded its functionality with hydra folder config, creating config files for model, trainer and web interface. For example, now we can just run python train_model.py –multirun trainer=trainer_cpu, trainer_gpu to test both cpu and gpu training using pytorch-lightning trainer configuration. The project is now easily expandable with new configurations just by adding a new config file to the proper folder and setting it to default if preferred.


### Question 13


> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:
> Ensuring the reproducibility of our experiments was integral to our methodology, accomplished through the use of configuration files managed by Hydra. When initiating an experiment, the hyperparameters from the specified config file, e.g. "config.yaml," were automatically incorporated into the model's training and prediction processes. These parameters could be dynamically modified from the command line, as detailed in the preceding question. With each run, Hydra not only saved the exact config file but also logged the printed output, preserving comprehensive information for future reference.
>
>This approach guarantees that no crucial details are overlooked, enhancing the reproducibility of previous experiments. Replicating an experiment is a straightforward process; one simply reruns the code, specifying the config file from a prior experiment: python train_model.py -cd path/to/config --config-name=config_experiment. This systematic utilization of config files and Hydra not only enhances the clarity and traceability of experiments but also streamlines the replication process, contributing to the reliability and transparency of our experimental outcomes.




### Question 14


> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:
> We employed W&B to monitor our model's training progress and performance. In our initial testing run, we tracked essential metrics such as train_loss, validation_loss, and epoch count. You can visualize these metrics in the following images:
> [epochs](figures/wandb_epochs.png)
> [train loss](figures/wandb_train_loss.png)
> [validation loss](figures/wandb_valid_loss.png)
>
> As depicted in these images, our focus was primarily on these core model values. However, it's noteworthy that our model's tracking capabilities are easily expandable. For instance, we can seamlessly extend it to include additional metrics like accuracy by incorporating logging in the PyTorch Lightning model class. This flexibility ensures that our monitoring can adapt and evolve as needed, providing a comprehensive understanding of the model's performance over time.




### Question 15


> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments? Include how you would run your docker images and include a link to one of your docker files.**
>
> Answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:
>In our experimentation, we established Docker images for diverse purposes, namely predict_model, train_model and web. This segmentation allowed us to encapsulate specific functionalities within each container, promoting modularity, flexibility and reproducibility.
>
> For example, to initiate our experiment, we can run the command: docker compose run --build. This command reads configurations from our docker-compose.yaml file, orchestrating the simultaneous launch of multiple Docker containers. This composition enables the seamless interaction of the "predict_model," "train_model," and "web" containers, streamlining the execution of our entire system.
>
> This approach simplifies deployment and scalability while maintaining a consistent environment across different stages of our project, contributing to a cohesive and efficient development workflow.
>
> Link to a docker file: [GCD docker image](https://console.cloud.google.com/artifacts/docker/peaceful-basis-411414/us-central1/artifact-repo/train-model/sha256:09133f871bd01adea230622f0f9f551135c33d273d8b2e4af097ec11c380a60b?project=peaceful-basis-411414)




### Question 16


> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:
> Debugging methods varied by members. Some used the error messages thrown by python exceptions, some used print statements to see desired outputs of algorithms and some used loggers. In the later stage of development, we tried to use mainly python logger for keeping track of important state changes in our project.


## Working in the cloud


> In the following section we would like to know more about your experience when developing in the cloud.


### Question 17


> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:
> The project makes use of multiple services from Google Cloud Platform (GCP), Google Cloud Build together with Google Artifact Registry, Google Bucket, Service Accounts, and Cloud Functions. 
For using these services/tools that Google Cloud offers, a Cloud project is created which allows to have the services/tools used under one project and thereby making it simpler to for example go from PR in git to build an docker image and run the training. 
> In the project, Google Cloud Build is used for building a docker image for training the model, this is done through a file ‘cloudbuild.yaml’, that is linked to the docker file ‘train_model.dockerfile’. The setup to trigger this step in the pipelines, comes from ‘git workflow’. The workflow is part of the CI pipeline and makes use of the file ‘submit_build.yaml’ to submit ‘cloudbuild.yaml’ to cloud, where Google Cloud Build then builds an docker image for training, triggered on both push and pull requests to the remote_storage and main branches.
> The same is true for building an image for prediction, where it follows the same steps. 
When the images are built, they are located in GCP ‘Artifact Registry’, where they can be managed. Because it’s fully integrated with Google Clouds tooling and runtimes, this makes it possible to automate pipelines through Google Cloud. 
>
> Google Bucket is used for storing both the raw and processed data together with the trained model also being saved to the bucket. 
> In the same way that git is used to work locally on code as an example, then DVC is the same for data, therefore when DVC is configured to point to the bucket as storage, the data can be kept updated and ensures everyone uses the same data.
> The DVC workflow follows the same as git, in that first data is local, but with the commands ‘dvc add’ and ‘dvc push’ data is pushed to the specified storage unit. To get the data locally , the command ‘dvc pull’ can be used, and the same goes for the trained mode. 
>
> In this project, the way the Bucket is used is two fold - in that it can be used locally but also in the cloud. 
> Locally the Buckets ensures that the user pulls the correct data to use for the project. While for the Cloud it’s part of the pipeline, in that the stored data in the Bucket is used for training the model inside a docker in the cloud. And also storing the trained model, so it can be used either locally or through API. 


### Question 18


> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:
> The Compute engine is used in combination with the Vertex AI, to train the model on a VM in the cloud and give the news model as output to the bucket. 
> The configuration is done in the config file ‘config_vertex_ai.yaml’, here the VM is specified to use n1-standard-8 which can be found here: https://cloud.google.com/compute/docs/general-purpose-machines#c3-highcpu_1, the specific model is this case is a machine with 8 vCPUs and 30GB memory.
> To monitor the performance of the training, the VM need access to where the weights and bias is located, which in this case is at WANDB.
> The machine takes the newly created docker image to spin up a container for the run. 
The training is then done here and the output is a trained model, that can be used later for prediction through a local FASTAPI.
>
> Command to run the Vertex AI:
gcloud ai custom-jobs create --region=europe-west1 --display-name=test-run --config=config_vertex_ai.yaml

### Question 19


> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:


--- question 19 fill here ---


### Question 20


> **Upload one image of your GCP container registry, such that we can see the different images that you have stored.**
> **You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:


--- question 20 fill here ---


### Question 21


> **Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:


--- question 21 fill here ---


### Question 22


> **Did you manage to deploy your model, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:
> For local deployment we utilized FastAPI, which accepts POST requests. For accessing this service, one can call `curl -X POST -F data=@/path/to/image`, or navigate to the /docs page on the local host. Here an image can be uploaded by the client, and then the server uses the latest trained model from the Cloud to predict the class of the uploaded image. For cloud deployment, we began setting up an API through Cloud Functions, however due to limited time we were unsuccessful in fully implementing it. 




### Question 23


> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:
> Unfortunately, we encountered challenges in incorporating model monitoring into our cloud infrastructure. Our objective is to establish a robust monitoring system that allows us to track various aspects of our models over time, including their effectiveness, usage patterns, and the occurrence of data drift. By harnessing these key metrics, we aim to enhance our ability to promptly identify areas that require attention and implement appropriate fixes and modifications to optimize overall performance.


### Question 24


> **How many credits did you end up using during the project and what service was most expensive?**
>
> Answer length: 25-100 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ...*
>
> Answer:
> Through GCD billing reports, our expenses were initially under $5, subsequently reduced to $0 due to a promotion. The primary allocation went to the Artifact Registry Network, accounting for approximately $2.5. Additional expenditures encompassed rented computational cores and VertexAI training pipelines, collectively totaling under $2.


## Overall discussion of project


> In the following section we would like you to think about the general structure of your project.


### Question 25


> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally in your own words, explain the**
> **overall steps in figure.**
>
> Answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to github, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:


--- question 25 fill here ---


### Question 26


> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:


	!!FILL!! - challenges: 
 	github for cooperation?
	dvc sync and secret api for pipeline?
	docker container built in github pipeline?
	Debugging for every single step on the road…
	


### Question 27


> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:
> s193396 - data loading, hydra experiment configuration, building and pushing Docker images to GCP, FastAPI implementation using trained model, Cloud deployment 
>
> s204165 - implementing deep learning model, continuous container, building and pushing Docker images to Docker Hub and GCP, Cloud setup and deployment
>
> s103629 -
>
> s240485 - data transformation using Albumentation and data loading pipeline, TIMM, pytorch-lightning, click integration, W&B logging, fastapi model service with docker container, hydra experiment multi-configuration
>
> s204127 - scaling of data creation to DVC, unit-testing, continuous integration of tests of GC data and model pulling, model, data and training test, Vertex AI training and test of Vertex AI prediction test in CI, wandb integration.
>
> All members contributed to code by fixing bugs and code mistakes, reviewing pull requests and solving merge conflicts.
