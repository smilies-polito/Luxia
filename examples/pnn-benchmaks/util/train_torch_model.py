from util.mem_usage import get_cpu_mem_usage, get_gpu_mem_usage
from util.utils import print_torch_model_summary
from util.utils import train_epoch, val_epoch
import wandb
import torch
import time
import os


def train_torch_model(model, datal_train, datal_val, args):
    data_lists = {
        "loss": [], "val_loss": [], "accuracy": [], "val_accuracy": [],
        "time": [], "cpu_mem": []}

    if args.device.type == 'cuda':
        data_lists['gpu_mem'] = []

    dacbool = False
    if args.dac_res is not None:
        dacbool = True
        data_lists['dac_qerror'] = []

    adcbool = False
    if args.adc_res is not None:
        adcbool = True
        data_lists['adc_qerror'] = []

    hetbool = False
    if args.heaters_res is not None:
        hetbool = True
        data_lists['heaters_qerror'] = []

    if args.save_model:
        torch.save(model, os.path.join(
            args.exp_dir, f"model_epoch{0:03d}.pth"))

    if args.verbose:
        print_torch_model_summary(model)

    print("[*] Define the loss function and optimizer")
    print()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=args.lr
    )

    if args.verbose:
        print(f"[*] Start training \t EPOCHS:{args.epoch} \t LR:{args.lr}"
              f"\t BATCH_SIZE:{args.batch_size} \t VAL_FRAC:{args.val_frac}"
              f"\t SEED:{args.seed}")
        print()

    improv_counter = 0
    improv_th = 10
    avg_vloss_best = 100_000_000

    for epoch in range(args.epoch):
        print(f'EPOCH {epoch+1}:')

        start = time.time()  # start counting time
        model.train()
        retvals = train_epoch(model=model,
                              optimizer=optimizer,
                              loss_fn=loss_fn,
                              device=args.device,
                              dataloader=datal_train,
                              is_qerror_het=hetbool,
                              is_qerror_dac=dacbool,
                              is_qerror_adc=adcbool
                              )
        avg_loss = retvals["loss"][0]
        avg_acc = retvals["acc"][0]
        if args.heaters_res is not None:
            avg_het_qerror = retvals["heaters_qerror"][0]
        if args.dac_res is not None:
            avg_dac_qerror = retvals["dac_qerror"][0]
        if args.adc_res is not None:
            avg_adc_qerror = retvals["adc_qerror"][0]

        # we don't need gradients to evaluate the model
        model.eval()
        avg_vloss, avg_vacc = val_epoch(model=model,
                                        loss_fn=loss_fn,
                                        device=args.device,
                                        dataloader=datal_val)

        deltat = time.time()-start  # end counting time
        cpuused_MB = get_cpu_mem_usage(args.verbose)

        if args.device.type == 'cuda':
            gpufree_MB, gpuused_MB, gputotal_MB = get_gpu_mem_usage(
                args.verbose)
            data_lists['gpu_mem'].append(gpuused_MB)

        data_lists["loss"].append(avg_loss)
        data_lists["val_loss"].append(avg_vloss)
        data_lists["accuracy"].append(avg_acc)
        data_lists["val_accuracy"].append(avg_vacc)
        data_lists["time"].append(deltat)
        data_lists["cpu_mem"].append(cpuused_MB)

        if args.dac_res is not None:
            data_lists['dac_qerror'].append(avg_dac_qerror)
        if args.adc_res is not None:
            data_lists['adc_qerror'].append(avg_adc_qerror)
        if args.heaters_res is not None:
            data_lists['heaters_qerror'].append(avg_het_qerror)

        if args.use_wandb:
            wandb_dict = {
                "train_loss": avg_loss,
                "val_loss": avg_vloss,
                "train_accuracy": avg_acc,
                "val_accuracy": avg_vacc,
                "time": deltat,
            }
            if args.dac_res is not None:
                wandb_dict["dac_qerror"] = avg_dac_qerror
            if args.adc_res is not None:
                wandb_dict["adc_qerror"] = avg_adc_qerror
            if args.heaters_res is not None:
                wandb_dict["heaters_qerror"] = avg_het_qerror
            wandb.log(wandb_dict)

        if args.verbose:
            print(f'LOSS train {avg_loss} valid {avg_vloss}')
            print(f'ACC train {avg_acc} valid {avg_vacc}')
            print(f'ELAPSED TIME {deltat} seconds')
            if hetbool:
                print(f'QERROR heaters {avg_het_qerror*100} %')
            if dacbool:
                print(f'QERROR DAC {avg_dac_qerror*100} %')
            if adcbool:
                print(f'QERROR ADC {avg_adc_qerror*100} %')
            print(flush=True)
            print()

        # Save the model if the validation loss is the best
        if args.save_model:
            if avg_vloss < avg_vloss_best:
                improv_counter += 1
                avg_vloss_best = avg_vloss
                if improv_counter >= improv_th:
                    print("Saving model...")
                    torch.save(
                        model,
                        os.path.join(
                            args.exp_dir, f"model_epoch{epoch+1:03d}.pth")
                    )
                    improv_counter = 0
    return data_lists
