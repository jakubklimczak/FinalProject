#!/bin/bash
# Script to download Kaggle competition dataset I use in my project.
USERNAME="jaklimczak"

usage() {
    echo "Usage: $0 [-u <username>]"
    echo "  -u, --username <username> : Specify the Kaggle username (default set to mine: jaklimczak)"
    echo "Before running the command, you need to install Kaggle CLI, execute: pip install kaggle"
    echo "Remember to obtain your own API key to access the Kaggle dataset. Follow the following tutorial if you don't know how: "
    echo "https://www.kaggle.com/docs/api"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -u|--username)
            shift
            USERNAME=$1
            ;;
        *)
            usage
            ;;
    esac
    shift
done

if [ "$(id -u)" -ne 0 ]; then
    echo "Please run as root (sudo)"
    exit 1
fi

export KAGGLE_CONFIG_DIR="/home/$USERNAME/.kaggle"
sudo -E -u "$USERNAME" /home/"$USERNAME"/.local/bin/kaggle competitions download -c rsna-miccai-brain-tumor-radiogenomic-classification -p ./dataset --force
