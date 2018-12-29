sudo apt-get update
sudo apt-get upgrade

ROOT_FOLDER = $PWD

bash ./install_ncsdk.sh
cd $ROOT_FOLDER
bash ./compile_appzoo.sh
cd $ROOT_FOLDER


