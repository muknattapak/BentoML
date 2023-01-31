# BentoML_Micro_service
Deploy Micro service through BentoML
## How to run this bentoml
1. clone project:
   - `git clone https://github.com/muknattapak/BentoML.git`
2. Go into bento folder for train the model first:
   - `pip install -r requirements.txt`
   - `python train.py`

   After trained the model. We can check the model in bento through:
   - `bentoml models list`
3. Build docker image
   - `bentoml build`
   - `bentoml list`
   - `bentoml containerize dt_latest2:latest`
   - `docker run -it --rm -p 3000:3000 dt_latest2:<name of Tag>`
4. Open API local host:
   - http://127.0.0.1:3000/
