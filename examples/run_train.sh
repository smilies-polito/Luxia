# # Clements training:
# # Description: Train the Clements accelerator on the Iris dataset
# python ./train.py --model=clements --dataset=iris --n_inputs=8 --n_layers=8 --n_networks=2 --epoch=50 --batch_size=12 --lr=1e-2
# # Description: Train the Clements accelerator on the Iris dataset with the is_matmul flag
python ./train.py --model=clements --dataset=iris --n_inputs=8 --n_layers=8 --n_networks=2 --epoch=50 --batch_size=12 --lr=1e-2 --is_matmul
#
# # Fldzhyan training 
# # Description: Train the Fldzhyan accelerator on the Iris dataset
# python ./train.py --model=fldzhyan --dataset=iris --n_inputs=8 --n_layers=8 --n_networks=2 --epoch=50 --batch_size=12 --lr=1e-2
# # Description: Train the Fldzhyan accelerator on the Iris dataset with the is_matmul flag
# python ./train.py --model=fldzhyan --dataset=iris --n_inputs=8 --n_layers=8 --n_networks=2 --epoch=50 --batch_size=12 --lr=1e-2 --is_matmul
# # Description: Train the Fldzhyan accelerator on the Iris dataset
# python ./train.py --model=fldzhyan_bell--dataset=iris --n_inputs=8 --n_layers=8 --n_networks=2 --epoch=50 --batch_size=12 --lr=1e-2
# # Description: Train the Fldzhyan bell accelerator on the Iris dataset with the is_matmul flag
# python ./train.py --model=fldzhyan_bell --dataset=iris --n_inputs=8 --n_layers=8 --n_networks=2 --epoch=50 --batch_size=12 --lr=1e-2 --is_matmul
#
# # Custom mesh
# # Description: Train the MyPhMesh accelerator on the Iris dataset
# python ./train.py --model=my_ph_mesh --dataset=iris --n_inputs=8 --n_layers=8 --n_networks=2 --epoch=50 --batch_size=12 --lr=1e-2
