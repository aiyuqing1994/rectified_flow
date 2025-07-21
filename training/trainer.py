import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from evaluation.fid import calculate_fid_for_model
from generation.generators import generate_and_save_comparison_images
from .loss import LPIPSHuberLoss, L2Loss
from .sampler import UShapedSampler, UniformSampler


def train_one_epoch(model, ema, dataloader, optimizer, loss_fn, device, stage, config):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Training Stage {stage}")
    sampler = UShapedSampler() if config.get("sampler") == "U-shape" else UniformSampler()

    for data in pbar:
        optimizer.zero_grad()

        if stage == 1:
            z1 = data[0].to(device)
            z0 = torch.randn_like(z1)
            t_scalar = sampler.sample((z0.shape[0],), device)
            t = t_scalar.view(-1, 1, 1, 1)
            zt = (1 - t) * z0 + t * z1
            v_target = z1 - z0
            v_pred = model(zt, t_scalar)
            if isinstance(loss_fn, LPIPSHuberLoss):
                x_pred = zt + (1 - t) * v_pred
                loss = loss_fn(v_pred, v_target, z1, x_pred, t_scalar)
            elif isinstance(loss_fn, L2Loss):
                loss = loss_fn(v_pred, v_target)
            else:
                raise f"Unsupported loss type {loss_fn}"
        elif stage == 2:
            z0, z1 = data
            z0 = z0.to(device)
            z1 = z1.to(device)
            t_scalar = sampler.sample((z0.shape[0],), device)
            t = t_scalar.view(-1, 1, 1, 1)
            zt = (1 - t) * z0 + t * z1
            v_target = z1 - z0
            v_pred = model(zt, t_scalar)
            if isinstance(loss_fn, LPIPSHuberLoss):
                x_pred = zt + (1 - t) * v_pred
                loss = loss_fn(v_pred, v_target, z1, x_pred, t_scalar)
            elif isinstance(loss_fn, L2Loss):
                loss = loss_fn(v_pred, v_target)
            else:
                raise f"Unsupported loss type {loss_fn}"
        else:  # stage == 3
            assert isinstance(loss_fn, L2Loss), "Stage 3 must use L2Loss for distillation."
            z0, z1 = data
            z0 = z0.to(device)
            z1 = z1.to(device)
            target = z1
            pred = model(z0)
            loss = loss_fn(pred, target)

        loss.backward()
        optimizer.step()
        ema.update()
        total_loss += loss.item()
        pbar.set_postfix({"Loss": loss.item()})
    return total_loss / len(dataloader)


def validate_one_epoch(model, dataloader, loss_fn, device, stage, config):
    model.eval()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Validating Stage {stage}")
    sampler = UShapedSampler() if config.get("sampler") == "U-shape" else UniformSampler()

    with torch.no_grad():
        for data in pbar:
            if stage == 1:
                z1 = data[0].to(device)
                z0 = torch.randn_like(z1)
                t_scalar = sampler.sample((z0.shape[0],), device)
                t = t_scalar.view(-1, 1, 1, 1)
                zt = (1 - t) * z0 + t * z1
                v_target = z1 - z0
                v_pred = model(zt, t_scalar)
                if isinstance(loss_fn, LPIPSHuberLoss):
                    x_pred = zt + (1 - t) * v_pred
                    loss = loss_fn(v_pred, v_target, z1, x_pred, t_scalar)
                elif isinstance(loss_fn, L2Loss):
                    loss = loss_fn(v_pred, v_target)
                else:
                    raise f"Unsupported loss type {loss_fn}"
            elif stage == 2:
                z0, z1 = data
                z0 = z0.to(device)
                z1 = z1.to(device)
                t_scalar = sampler.sample((z0.shape[0],), device)
                t = t_scalar.view(-1, 1, 1, 1)
                zt = (1 - t) * z0 + t * z1
                v_target = z1 - z0
                v_pred = model(zt, t_scalar)
                if isinstance(loss_fn, LPIPSHuberLoss):
                    x_pred = zt + (1 - t) * v_pred
                    loss = loss_fn(v_pred, v_target, z1, x_pred, t_scalar)
                elif isinstance(loss_fn, L2Loss):
                    loss = loss_fn(v_pred, v_target)
                else:
                    raise f"Unsupported loss type {loss_fn}"
            else:  # stage == 3
                assert isinstance(loss_fn, L2Loss), "Stage 3 must use L2Loss for distillation."
                z0, z1 = data
                z0 = z0.to(device)
                z1 = z1.to(device)
                target = z1
                pred = model(z0)
                loss = loss_fn(pred, target)
            total_loss += loss.item()
            pbar.set_postfix({"Loss": loss.item()})
    return total_loss / len(dataloader)


def train_stage(config, stage_name, model, ema, train_loader, val_loader, optimizer, loss_fn, device, models_save_dir, images_save_dir, mu_real, sigma_real, inception_model, writer):
    print(f"--- {stage_name}: Training --- ")
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    stage_num = int(stage_name.split(' ')[-1])
    epochs = config[f"epochs_stage{stage_num}"]

    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, ema, train_loader, optimizer, loss_fn, device, stage=stage_num, config=config)
        writer.add_scalar(f"Loss/{stage_name}", avg_loss, epoch)
        print(f"{stage_name} | Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        val_loss = validate_one_epoch(ema.ema_model, val_loader, loss_fn, device, stage=stage_num, config=config)

        writer.add_scalar(f"Loss/val_{stage_name}", val_loss, epoch)
        print(f"{stage_name} | Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}")
        scheduler.step(avg_loss)
        generate_and_save_comparison_images(ema.ema_model, config, device, epoch, stage_num, images_save_dir, writer)
        if (epoch + 1) % config['checkpoint_interval'] == 0:
            torch.save({'model_state_dict': model.state_dict(), 'ema_state_dict': ema.state_dict()}, f"{models_save_dir}/stage{stage_num}_epoch_{epoch+1}.pth")
            if config["calculate_fid"]:
                fid_score = calculate_fid_for_model(ema.ema_model, mu_real, sigma_real, inception_model, config, device, stage_num)
                writer.add_scalar(f"FID/{stage_name}", fid_score, epoch)
                print(f"FID Score after {epoch+1} epochs: {fid_score:.2f}")

    print(f"--- {stage_name} Finished ---")
    model.eval()
