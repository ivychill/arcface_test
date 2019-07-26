#!/bin/bash
help_info ()
{
	echo "****************************************************************************"
	echo "*"
	echo "* MODULE:             Linux Script - facenet"
	echo "*"
	echo "* COMPONENT:          This script used to run facenet python process"
	echo "*"
	echo "* REVISION:           $Revision: 2.0 $"
	echo "*"
	echo "* DATED:              $Date: 2019-07-05 15:16:28 +0000 () $"
	echo "*"
	echo "* AUTHOR:             kang.liu"
	echo "*"
	echo "***************************************************************************"
	echo ""
	echo "* Copyright kang.liu@kuang-chi.com. 2020. All rights reserved"
	echo "*"
	echo "***************************************************************************"
}

case $1 in
    represent|s)
        export PYTHONPATH=$(pwd)/src
        echo "********get  represent********"
        python src/represent.py \
        --data_path /data/fengchen/ensemble/dataset/helmet/helmet_112 \
        --output_dir $(pwd)/Saveout \
        --det  1 \
        --gpu  3 \
    ;;

    result)
        export PYTHONPATH=$(pwd)/src
        python src/mxnet_test_model_plot_roc.py\
        --feature_path $(pwd)/Saveout\
        --output_dir $(pwd)/Saveout\
        ;;

    clear)
        find . -name "*.pyc" -type f -print -exec rm -rf {} \;
    ;;

    *)
		help_info
		exit 1
    ;;
esac
exit 0