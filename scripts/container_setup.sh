#! /bin/bash

apt update
apt install git

# Change username and email if needed
git config --global user.name "RafikHachana"
git config --global user.email "rafikhachana@gmail.com"

git config --global credential.helper 'cache --timeout=7200'

git clone https://github.com/RafikHachana/Sym-JEPA.git

