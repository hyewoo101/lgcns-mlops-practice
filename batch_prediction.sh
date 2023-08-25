BASH_ENV=~/.bashrc
ROOT_PATH=/workspaces/lgcns-mlops-practice
PIPENV_PIPFILE=$ROOT_PATH/Pipfile

export PATH=$PATH:/usr/local/py-utils/bin # 이 환경 변수를 추가시켜줘야 pipenv를 잡을 수 있음
export PIPENV_PIPFILE=$PIPENV_PIPFILE
pipenv run python $ROOT_PATH/batch_prediction.py >> $ROOT_PATH/cron.log 2>&1 # 실행시키고 log는 여기다 쌓아줘
