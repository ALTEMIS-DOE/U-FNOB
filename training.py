# %%
import torch
import numpy as np
from U_FNOB import *
from U_FNOB import *
from lploss import *
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tqdm import tqdm
import torch.utils.data as data
import gc

torch.manual_seed(0)
np.random.seed(0)

import wandb
wandb.init(project="PSSM", entity="fdl-digitaltwin")

def main(epochs, batch_size, learning_rate, ufno_model, UNet, beta1, beta2, beta3, beta4, beta5, beta6, beta7, dataset = 'even_interval'):
    
    # dataset = 'even_interval', dataset = 'uneven_interval'
    
    # Empty cache before starting
    gc.collect()
    torch.cuda.empty_cache()
        
    # PARAMETERS
    e_start = 0
    scheduler_step = 4
    scheduler_gamma = 0.85
    
    
    
    data_path = 'us-digitaltwiner-pub-features/srs_farea_ensemble_simulations_dataset/'
    if dataset == 'even_interval':
        f_input = BytesIO(file_io.read_file_to_string("gs://" + data_path + 'input_recurrent.npy', binary_mode=True)) 
        f_output = BytesIO(file_io.read_file_to_string("gs://" + data_path + 'output_recurrent.npy', binary_mode=True))
    elif dataset == 'uneven_interval':
        f_input = BytesIO(file_io.read_file_to_string("gs://" + data_path + 'input_top_layer.npy', binary_mode=True)) 
        f_output = BytesIO(file_io.read_file_to_string("gs://" + data_path + 'output.npy', binary_mode=True))
  
    input_array = torch.from_numpy(np.load(f_input)) 
    output_array = torch.from_numpy(np.load(f_output))
    
    # size of array from the input
    ns, nz, nx, nt, nc = input_array.shape
    no = output_array.shape[-1]
    nc = nc - 3

    # meta_data
    if dataset == 'even_interval':
        f = (BytesIO(file_io.read_file_to_string("gs://" + data_path + 'meta_data_recurrent.txt', binary_mode=True)))
    elif dataset == 'uneven_interval':
        f = (BytesIO(file_io.read_file_to_string("gs://" + data_path + 'meta_data.txt', binary_mode=True)))
        
    lines = f.readlines()
    input_names = str(lines[0]).split('\'')[1].split('\\n')[0].split(', ')
    time_steps = np.array(str(lines[1]).split('\'')[1].split('\\n')[0].split(', '),dtype = 'float64')
    time_steps = np.array(time_steps, dtype = 'int64')
    input_min = np.array(str(lines[2]).split('\'')[1].split('\\n')[0].split(', '),dtype = 'float64')
    input_max = np.array(str(lines[3]).split('\'')[1].split('\\n')[0].split(', '),dtype = 'float64')
    output_names = str(lines[4]).split('\'')[1].split('\\n')[0].split(', ')
    
    # rescale output
    tritium_MCL = 7e-13
    # Custom min and max values per variable for rescaling
    rescale_factors = {
        0 : {
            'min': np.nanmin(output_array[:,:,:,:,0]),
            'max': np.nanmax(output_array[:,:,:,:,0])/2
        },
        1 : {
            'min': np.nanmin(output_array[:,:,:,:,1]),
            'max': np.nanmax(output_array[:,:,:,:,1])/5
        },
        2 : {
            'min': np.nanmin(output_array[:,:,:,:,2]),
            'max': np.nanmax(output_array[:,:,:,:,2])
        },
        3 : {
            'min': np.nanmin(output_array[:,:,:,:,3]),
            'max': np.nanmax(output_array[:,:,:,:,3])
        },
        4 : {
            'min': np.nanmin(output_array[:,:,:,:,4]),
            'max': np.nanmax(output_array[:,:,:,:,4])
        },
        5 : {
            'min': tritium_MCL*0.2,
            'max': 9e-9
        },
        6 : {
            'min': np.nanmin(output_array[:,:,:,:,6]),
            'max': np.nanmax(output_array[:,:,:,:,6])
        }
    }
    
    
    # Rescale input
    input_max_values = np.nanmax(input_array.reshape(-1,nc+3),axis = 0).reshape(1,1,1,1,-1)
    input_array = input_array/input_max_values

    # Input nan -> 0
    input_array[np.isnan(input_array)] = 0

    # Rescale output_array between 0 and 1.

    scaled_output = output_array.detach().clone()

    for i in range(no):
        scaled_output[:,:,:,:,i][scaled_output[:,:,:,:,i]<rescale_factors[i]['min']] = rescale_factors[i]['min']
        scaled_output[:,:,:,:,i][scaled_output[:,:,:,:,i]>rescale_factors[i]['max']] = rescale_factors[i]['max']
        scaled_output[:,:,:,:,i] = (scaled_output[:,:,:,:,i] - rescale_factors[i]['min'])/(rescale_factors[i]['max']-rescale_factors[i]['min'])

    scaled_output[np.isnan(scaled_output)] = 0
    
    # Current training
    selected_idx = np.array([0,1,2,5])
    scaled_output_4 = scaled_output[:,:,:,:,selected_idx]
    output_names_4 = list(np.array(output_names)[selected_idx])
    
    # Build U-FNO model

    # %%
    mode1 = 10
    mode2 = 10
    mode3 = 4
    width = 36
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if ufno_model == 'UFNOBR':
        if UNet:
            model = UFNOBR(mode1, mode2, width, UNet = True)
            model_head = UFNOBR(mode1, mode2, width, UNet = True)
            #model_head = torch.load('final_models/UFNOBR_model_head_best_combination_200epochs')
            #model = torch.load('final_models/UFNOBR_model_best_combination_150epochs')
        else:
            model = UFNOBR(mode1, mode2, width, UNet = False)
            model_head = UFNOBR(mode1, mode2, width, UNet = False)
    elif ufno_model == 'UFNOB':
        if UNet: 
            model = UFNOB(mode1, mode2, mode3, width, UNet = True)
            #model = torch.load('final_models/UFNOUFNOB_model_best_combination_150epochs')
        else:
            model = UFNOB(mode1, mode2, mode3, width, UNet = False)

    model.to(device)
    
    if ufno_model == 'UFNOBR':
        model_head.to(device)
        
    # prepare derivatives
    
    grid_x = input_array[0,8,:,0,-3]
    grid_dx =  - grid_x[:-2] + grid_x[2:]
    grid_dx = grid_dx[None, None, :, None, None].to(device)

    grid_z = input_array[0,:,:,0,-2]
    grid_dz =  - grid_z[:-2,:] + grid_z[2:,:] 
    grid_dz[grid_dz==0] = 1/nz # to avoid divide by 0
    grid_dz = grid_dz[None, :, :, None, None].to(device)

    # bottom_z location
    bottom_z = np.zeros(nx)
    for idx_x in range(nx):
        nan_idx = np.where(np.isnan(output_array[0,:10,idx_x,0,0])==1)[0]
        if len(nan_idx)>0:
            bottom_z[idx_x] = np.max(nan_idx)+1
        else:
            bottom_z[idx_x] = 0
    bottom_z = np.array(bottom_z,dtype = 'float64')

    wandb.config = {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "e_start": e_start,
        "scheduler_step": scheduler_step,
        "scheduler_gamma": scheduler_gamma,
        "model": ufno_model,
        "UNet": UNet,
        "beta1":beta1,
        "beta2":beta2,
        "beta3":beta3,
        "beta4":beta4,
        "beta5":beta5,
        "beta6":beta6,
        "beta7":beta7,
        "dataset":dataset
    }
    wandb.init(config=wandb.config)

   
    # Split dataset into training, val and test set
    
    torch_dataset = torch.utils.data.TensorDataset(input_array, scaled_output_4)

    dataset_sizes = [np.int(np.int(ns*0.8)/batch_size)*batch_size, np.int((ns-np.int(np.int(ns*0.8)/batch_size)*batch_size)/2),np.int((ns-np.int(np.int(ns*0.8)/batch_size)*batch_size)/2)]

    train_data, val_data, test_data = data.random_split(torch_dataset, dataset_sizes ,generator=torch.Generator().manual_seed(0))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader  = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    myloss = LpLoss(size_average=True) # relative lp loss
    
    if ufno_model == 'UFNOBR':
        optimizer_head = torch.optim.Adam(model_head.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler_head = torch.optim.lr_scheduler.StepLR(optimizer_head, step_size=scheduler_step, gamma=scheduler_gamma)
    
    # loss functions
    def loss_function(x,y, model, beta1, beta2):
        no = y.shape[-1]
        current_ns = x.shape[0]

        if len(x.shape)==5: # UFNOB
            nt = x.shape[-2]
        else: # UFNOBR
            nt = 1

        nz = x.shape[1]
        nx = x.shape[2]

        if len(x.shape)<5:
            mask = (x[:,:,:,0]!=0).reshape(x.shape[0], nz, nx, 1, 1).repeat(1,1,1,nt,no) 
        elif len(x.shape)==5:   
            mask = (x[:,:,:,0:1,0]!=0).reshape(x.shape[0], nz, nx, 1, 1).repeat(1,1,1,nt,no) # deactivate those input values with 0, i.e. above the surface
        dy_dx = (y[:,:,2:,:,:] - y[:,:,:-2,:,:])/grid_dx
        dy_dz = (y[:,2:,:,:,:] - y[:,:-2,:,:,:])/grid_dz

        pred = model(x.float()).view(-1, nz, nx, nt, no)

        ori_loss = 0
        der_x_loss = 0
        der_z_loss = 0


        # original loss
        for i in range(current_ns):
            ori_loss += myloss(pred[i,...][mask[i,...]].reshape(1, -1), y[i,...][mask[i,...]].reshape(1, -1))

        # 1st derivative loss
        # dx
        dy_pred_dx = (pred[:,:,2:,:,:] - pred[:,:,:-2,:,:])/grid_dx
        mask_dy_dx = mask[:,:,:(nx-2),:,:]

        for i in range(current_ns):
            der_x_loss += myloss(dy_pred_dx[i,...][mask_dy_dx[i,...]].reshape(1, -1), dy_dx[i,...][mask_dy_dx[i,...]].view(1, -1))


        # 1st derivative loss
        # dz
        dy_pred_dz = (pred[:,2:,:,:,:] - pred[:,:-2,:,:,:])/grid_dz
        mask_dy_dz = mask[:,:(nz-2),:,:,:]

        for i in range(current_ns):
            der_z_loss += myloss(dy_pred_dz[i,...][mask_dy_dz[i,...]].reshape(1, -1), dy_dz[i,...][mask_dy_dz[i,...]].view(1, -1))
        
        mre_loss = ori_loss#/current_ns
        der_loss_x = beta1 * der_x_loss
        der_loss_z = beta2 * der_z_loss
        return mre_loss, der_loss_x, der_loss_z


    def loss_function_boundary(x, y,  model, beta3, beta4, axis=3):

        # This is for the plume part
        # y should be the plume slice
        current_ns = x.shape[0]

        if len(x.shape)==5: # UFNOB
            nt = x.shape[-2]
        else: #UFNOBR
            nt = 1

        nz = x.shape[1]
        nx = x.shape[2]

        if len(x.shape)<5: #UFNOBR
            mask = (x[:,:,:,0]!=0).reshape(-1,nz,nx,1).repeat(1,1,1,nt) 
            #mask[:,:,:120,:] = torch.tensor(False, dtype=torch.bool) # We don't care the boundary on the left side (no plumes for now), for generalization please delete this line
        elif len(x.shape)==5:   
            mask = (x[:,:,:,0:1,0]!=0).repeat(1,1,1,nt) # deactivate those input values with 0, i.e. above the surface
            #mask[:,:,:120,:] = torch.tensor(False, dtype=torch.bool) # We don't care the boundary on the left side (no plumes for now), for generalization please delete this line

        MCL_threshold = (tritium_MCL-rescale_factors[selected_idx[axis]]['min'])/(rescale_factors[selected_idx[axis]]['max']-rescale_factors[selected_idx[axis]]['min'])
        y = (y>MCL_threshold)*1

        dy_dx = (y[:,:,2:,:] - y[:,:,:-2,:])/grid_dx[:,:,:,:,0]
        dy_dz = (y[:,2:,:,:] - y[:,:-2,:,:])/grid_dz[:,:,:,:,0]

        pred = model(x.float()).view(-1, nz, nx, nt, 4)[:,:,:,:,axis]
        pred = (pred>MCL_threshold)*1

        der_x_loss = 0
        der_z_loss = 0

        # 1st derivative loss
        # dx
        dy_pred_dx = (pred[:,:,2:,:] - pred[:,:,:-2,:])/grid_dx[:,:,:,:,0]
        #mask_dy_dx = mask[:,:,:(nx-2),:]
        mask_dy_dx = torch.clone(mask[:,:,:(nx-2),:])
        mask_dy_dx[dy_dx==0] =  torch.tensor(False, dtype=torch.bool)
        
        for i in range(current_ns):
            if (dy_dx[i,...]!=0).sum()!=0:
                der_x_loss += myloss(dy_pred_dx[i,...][mask_dy_dx[i,...]].reshape(1, -1), dy_dx[i,...][mask_dy_dx[i,...]].view(1, -1))

        # 1st derivative loss
        # dz
        dy_pred_dz = (pred[:,2:,:,:] - pred[:,:-2,:,:])/grid_dz[:,:,:,:,0]
        #mask_dy_dz = mask[:,:(nz-2),:,:]
        mask_dy_dz = torch.clone(mask[:,:(nz-2),:,:])
        mask_dy_dz[dy_dz==0] =  torch.tensor(False, dtype=torch.bool)
        
        for i in range(current_ns):
            if (dy_dz[i,...]!=0).sum()!=0:
                der_z_loss += myloss(dy_pred_dz[i,...][mask_dy_dz[i,...]].reshape(1, -1), dy_dz[i,...][mask_dy_dz[i,...]].view(1, -1))

        der_loss_x = beta3 * der_x_loss
        der_loss_z = beta4 * der_z_loss
        return der_loss_x, der_loss_z

    def loss_function_PINN_BC1(x, model, axis = 1): 

        # This is for darcy 1, z direction
        current_ns = x.shape[0]
        if len(x.shape)==5: # UFNOB
            nt = x.shape[-2]
        else: #UFNOBR
            nt = 1

        nz = x.shape[1]
        nx = x.shape[2]

        zero_value = (0-rescale_factors[selected_idx[axis]]['min'])/(rescale_factors[selected_idx[axis]]['max']-rescale_factors[selected_idx[axis]]['min'])

        if len(x.shape)<5:
            mask = (x[:,:,:,0]!=0).reshape(-1,nz,nx,1).repeat(1,1,1,nt) 
        elif len(x.shape)==5:   
            mask = (x[:,:,:,0:1,0]!=0).repeat(1,1,1,nt) # deactivate those input values with 0, i.e. above the surface

        pred = model(x.float()).view(-1, nz, nx, nt, 4)
        pred = pred[:,:,:,:,axis]

        pinn_BC_loss = 0

        # boundary condition loss
        for i in range(current_ns):
            pinn_BC_loss += torch.norm(torch.abs(pred[i,:,-1,:][mask[i,:,-1,:]]-zero_value),2)/nz # right x boundary
            pinn_BC_loss += torch.norm(torch.abs(pred[i,:,0,:][mask[i,:,0,:]]-zero_value),2)/nz# left x boundary       
        return pinn_BC_loss/(current_ns*2)


    def loss_function_PINN_BC2(x,  model, axis = 0): 
        # This is for darcy 0, x direction
        current_ns = x.shape[0]
        if len(x.shape)==5: # UFNOB
            nt = x.shape[-2]
        else: #UFNOBR
            nt = 1

        nz = x.shape[1]
        nx = x.shape[2]

        zero_value = (0-rescale_factors[selected_idx[axis]]['min'])/(rescale_factors[selected_idx[axis]]['max']-rescale_factors[selected_idx[axis]]['min'])

        if len(x.shape)<5:
            mask = (x[:,:,:,0]!=0).reshape(-1,nz,nx,1).repeat(1,1,1,nt) 
        elif len(x.shape)==5:   
            mask = (x[:,:,:,0:1,0]!=0).repeat(1,1,1,nt) # deactivate those input values with 0, i.e. above the surface

        pred = model(x.float()).view(-1, nz, nx, nt, 4)
        pred = pred[:,:,:,:,axis]

        pinn_BC_loss = 0

        # boundary condition loss
        for i in range(current_ns):
            pinn_BC_loss += torch.norm(torch.abs(pred[i,bottom_z,np.arange(nx),:][mask[i,bottom_z,np.arange(nx),:]]-zero_value),2)/nx # bottom z boundary

        return pinn_BC_loss/(current_ns)


    def loss_function_PINN_BC3(x, model, axis = 2): 
        # This is for hydraulic head 
        current_ns = x.shape[0]

        sx = 10
        sz = 2.5

        if len(x.shape)==5: # UFNOB
            nt = x.shape[-2]
        else: #UFNOBR
            nt = 1

        nz = x.shape[1]
        nx = x.shape[2]

        if len(x.shape)<5:
            mask = (x[:,:,:,0]!=0).reshape(-1,nz,nx,1).repeat(1,1,1,nt) 
        elif len(x.shape)==5:   
            mask = (x[:,:,:,0:1,0]!=0).repeat(1,1,1,nt) # deactivate those input values with 0, i.e. above the surface

        pred = model(x.float()).view(-1, nz, nx, nt, 4)
        pred = pred[:,:,:,:,axis]

        pinn_BC_loss = 0

        # boundary condition loss
        for i in range(current_ns):
            pinn_BC_loss += torch.norm(torch.abs((pred[i,:,-2,:]-pred[i,:,-1,:])/sx)[mask[i,:,-1,:]],2)/nz # right x boundary
            pinn_BC_loss += torch.norm(torch.abs((pred[i,:,1,:]- pred[i,:,0,:])/sx)[mask[i,:,0,:]],2)/nz# left x boundary
            pinn_BC_loss += torch.norm(torch.abs((pred[i,bottom_z+1,np.arange(nx),:]-pred[i,bottom_z,np.arange(nx),:])/sz)[mask[i,bottom_z,np.arange(nx),:]],2)/nx # bottom z boundary

        return pinn_BC_loss/(current_ns*3)


    # Training
    def training_loop_UFNOBR(model_current = 'model_head',epochs = 100):

        plume_axis = np.where(np.array(output_names_4) == "total_component_concentration.cell.Tritium conc")[0][0]
        darcy_x_axis = np.where(np.array(output_names_4) == "darcy_velocity.cell.0")[0][0]
        darcy_z_axis = np.where(np.array(output_names_4) == "hydraulic_head.cell.0")[0][0]
        hh_axis = np.where(np.array(output_names_4) == "hydraulic_head.cell.0")[0][0]

        train_l2 = 0.0
        train_loss_array = np.zeros(epochs,)
        val_loss_array = np.zeros(epochs)

        im_init = torch.mean(scaled_output_4[:,:,:,0,:],axis = 0).to(device)

        #mask = (scaled_output_4[0:1,:,:,0:1,0]!=0).reshape(1, nz, nx, 1).repeat(batch_size,1,1,4)

        for ep in range(1,epochs+1):
            if model_current == 'model_head':
                model_head.train()
                num_time = 1
            else:
                model.train()
                num_time = nt

            train_l2 = 0
            val_l2 = 0
            counter = 0

            for xx, yy in train_loader:

                im = im_init.repeat(xx.shape[0],1,1,1) 

                num_current_batch = xx.shape[0]

                loss = 0
                loss_array = torch.zeros(8)

                xx = xx.to(device)
                yy = yy.to(device)

                for t in np.arange(num_time):
                    x = torch.cat((xx[:,:,:,t,:], im), dim=-1).to(device)

                    y = yy[:,:,:,t:(t+1),:]

                    if t == 0:
                        im = model_head(x.float())
                        mre_loss, der_loss_x, der_loss_z = loss_function(x,y, model_head, beta1, beta2)
                        plume_loss_x, plume_loss_z = loss_function_boundary(x, y[:,:,:,:,plume_axis],model_head, beta3, beta4, axis=plume_axis)
                        PINN_loss_darcy_z = beta5*loss_function_PINN_BC1(x,model_head, axis=darcy_z_axis)
                        PINN_loss_darcy_x = beta6*loss_function_PINN_BC2(x,model_head, axis=darcy_x_axis)
                        PINN_loss_hydraulic_head = beta7*loss_function_PINN_BC3(x,model_head,axis=hh_axis)
                        loss = loss +  mre_loss + der_loss_x + der_loss_z + plume_loss_x + plume_loss_z+ PINN_loss_darcy_z + PINN_loss_darcy_x + PINN_loss_hydraulic_head
                        loss_array = loss_array + torch.tensor([mre_loss,der_loss_x,der_loss_z,plume_loss_x,plume_loss_z,PINN_loss_darcy_z,PINN_loss_darcy_x,PINN_loss_hydraulic_head])
                        pred = im
                    else:
                        im = model(x.float())
                        mre_loss, der_loss_x, der_loss_z = loss_function(x,y, model, beta1, beta2)
                        plume_loss_x, plume_loss_z = loss_function_boundary(x, y[:,:,:,:,plume_axis],model, beta3, beta4, axis=plume_axis)
                        PINN_loss_darcy_z = beta5*loss_function_PINN_BC1(x,model, axis=darcy_z_axis)
                        PINN_loss_darcy_x = beta6*loss_function_PINN_BC2(x,model, axis=darcy_x_axis)
                        PINN_loss_hydraulic_head = beta7*loss_function_PINN_BC3(x,model,axis=hh_axis)
                        loss = loss +  mre_loss + der_loss_x + der_loss_z + plume_loss_x + plume_loss_z+ PINN_loss_darcy_z + PINN_loss_darcy_x + PINN_loss_hydraulic_head
                        loss_array = loss_array + torch.tensor([mre_loss,der_loss_x,der_loss_z,plume_loss_x,plume_loss_z,PINN_loss_darcy_z, PINN_loss_darcy_x,PINN_loss_hydraulic_head])
                        pred = torch.cat((pred, im), -1)

                if model_current == 'model_head':
                    optimizer_head.zero_grad()
                else:
                    optimizer.zero_grad()

                loss.backward()

                if model_current == 'model_head':
                    optimizer_head.step()   
                else:
                    optimizer.step()       

                counter += 1
                if counter % 100 == 0:
                    print(f'epoch: {ep}, batch: {counter}/{len(train_loader)}, train loss: {loss.item()/batch_size:.4f}')

                train_l2 += loss_array

            for xx, yy in val_loader:
                im = im_init.repeat(xx.shape[0],1,1,1) 

                num_current_batch = xx.shape[0]

                loss = 0
                loss_array = torch.zeros(8)
                
                xx = xx.to(device)
                yy = yy.to(device)

                for t in np.arange(num_time):

                    x = torch.cat((xx[:,:,:,t,:], im), dim=-1).to(device)

                    y = yy[:,:,:,t:(t+1),:]

                    if t == 0:
                        im = model_head(x.float())
                        mre_loss, der_loss_x, der_loss_z = loss_function(x,y, model_head, beta1, beta2)
                        plume_loss_x, plume_loss_z = loss_function_boundary(x, y[:,:,:,:,plume_axis],model_head, beta3, beta4, axis=plume_axis)
                        PINN_loss_darcy_z = beta5*loss_function_PINN_BC1(x,model_head, axis=darcy_z_axis)
                        PINN_loss_darcy_x = beta6*loss_function_PINN_BC2(x,model_head, axis=darcy_x_axis)
                        PINN_loss_hydraulic_head = beta7*loss_function_PINN_BC3(x,model_head,axis=hh_axis)
                        loss = loss +  mre_loss + der_loss_x + der_loss_z + plume_loss_x + plume_loss_z+ PINN_loss_darcy_z + PINN_loss_darcy_x + PINN_loss_hydraulic_head
                        loss_array = loss_array + torch.tensor([mre_loss,der_loss_x,der_loss_z,plume_loss_x,plume_loss_z,PINN_loss_darcy_z,PINN_loss_darcy_x,PINN_loss_hydraulic_head])
                        pred = im
                    else:
                        im = model(x.float())
                        mre_loss, der_loss_x, der_loss_z = loss_function(x,y, model, beta1, beta2)
                        plume_loss_x, plume_loss_z = loss_function_boundary(x, y[:,:,:,:,plume_axis],model, beta3, beta4, axis=plume_axis)
                        PINN_loss_darcy_z = beta5*loss_function_PINN_BC1(x,model, axis=darcy_z_axis)
                        PINN_loss_darcy_x = beta6*loss_function_PINN_BC2(x,model, axis=darcy_x_axis)
                        PINN_loss_hydraulic_head = beta7*loss_function_PINN_BC3(x,model,axis=hh_axis)
                        loss = loss +  mre_loss + der_loss_x + der_loss_z + plume_loss_x + plume_loss_z+ PINN_loss_darcy_z + PINN_loss_darcy_x + PINN_loss_hydraulic_head
                        loss_array = loss_array + torch.tensor([mre_loss,der_loss_x,der_loss_z,plume_loss_x,plume_loss_z,PINN_loss_darcy_z,PINN_loss_darcy_x,PINN_loss_hydraulic_head])
                        pred = torch.cat((pred, im), -1)

                val_l2 += loss_array

            train_loss = train_l2/dataset_sizes[0]
            val_loss = val_l2/dataset_sizes[1]
            print(f'epoch: {ep}, train loss: {train_loss}')
            print(f'epoch: {ep}, val loss:   {val_loss}')

            #train_loss_array[ep-1] = train_loss
           # val_loss_array[ep-1] = val_loss

            if model_current == 'model_head':
                scheduler_head.step()   
            else:
                scheduler.step()    

            lr_ = optimizer.param_groups[0]['lr']
            if ep % 5 == 0:
                PATH = f'saved_models/dP_UFNOBR_UNet_{UNet}_{ep}ep_{width}width_{mode1}m1_{mode2}m2_{input_array.shape[0]}model_{model_current}train_{lr_:.2e}lr'
                if model_current == 'model_head':
                    torch.save(model_head, PATH)
                else:
                    torch.save(model, PATH)

            # Framework agnostic / custom metrics
            wandb.log({"epoch": ep, "loss": train_loss.sum(), "val_loss": val_loss.sum()})
                    
        
    def training_loop_UFNOB():

        plume_axis = np.where(np.array(output_names_4) == "total_component_concentration.cell.Tritium conc")[0][0]
        darcy_x_axis = np.where(np.array(output_names_4) == "darcy_velocity.cell.0")[0][0]
        darcy_z_axis = np.where(np.array(output_names_4) == "hydraulic_head.cell.0")[0][0]
        hh_axis = np.where(np.array(output_names_4) == "hydraulic_head.cell.0")[0][0]

        train_loss_array = np.zeros(epochs)
        val_loss_array = np.zeros(epochs)

        for ep in range(1,epochs+1):
            model.train()
            train_l2 = 0
            val_l2 = 0
            counter = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                
                mre_loss, der_loss_x, der_loss_z = loss_function(x,y, model, beta1, beta2)
                plume_loss_x, plume_loss_z = loss_function_boundary(x, y[:,:,:,:,plume_axis],model, beta3, beta4, axis=plume_axis)
                PINN_loss_darcy_z = beta5*loss_function_PINN_BC1(x,model, axis=darcy_z_axis)
                PINN_loss_darcy_x = beta6*loss_function_PINN_BC2(x,model, axis=darcy_x_axis)
                PINN_loss_hydraulic_head = beta7*loss_function_PINN_BC3(x,model,axis=hh_axis)
                loss =  mre_loss + der_loss_x + der_loss_z + plume_loss_x + plume_loss_z+ PINN_loss_darcy_z + PINN_loss_darcy_x + PINN_loss_hydraulic_head
                loss_array = torch.tensor([mre_loss,der_loss_x,der_loss_z,plume_loss_x,plume_loss_z,PINN_loss_darcy_z,PINN_loss_darcy_x,PINN_loss_hydraulic_head])

                
                loss.backward()
                optimizer.step()
                train_l2 += loss_array
                
                counter += 1
                if counter % 100 == 0:
                    print(f'epoch: {ep}, batch: {counter}/{len(train_loader)}, train loss: {loss.item()/batch_size:.4f}')
                #print(loss_array)
                
            scheduler.step()

            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                mre_loss, der_loss_x, der_loss_z = loss_function(x,y, model, beta1, beta2)
                plume_loss_x, plume_loss_z = loss_function_boundary(x, y[:,:,:,:,plume_axis],model, beta3, beta4, axis=plume_axis)
                PINN_loss_darcy_z = beta5*loss_function_PINN_BC1(x,model, axis=darcy_z_axis)
                PINN_loss_darcy_x = beta6*loss_function_PINN_BC2(x,model, axis=darcy_x_axis)
                PINN_loss_hydraulic_head = beta7*loss_function_PINN_BC3(x,model,axis=hh_axis)
                loss =  mre_loss + der_loss_x + der_loss_z + plume_loss_x + plume_loss_z+ PINN_loss_darcy_z + PINN_loss_darcy_x + PINN_loss_hydraulic_head
                loss_array = torch.tensor([mre_loss,der_loss_x,der_loss_z,plume_loss_x,plume_loss_z,PINN_loss_darcy_z,PINN_loss_darcy_x,PINN_loss_hydraulic_head])
                
                val_l2 += loss_array
                
            train_loss = train_l2/dataset_sizes[0]
            val_loss = val_l2/dataset_sizes[1]
            print(f'epoch: {ep}, train loss: {train_loss}, sum: {train_loss.sum()}')
            print(f'epoch: {ep}, val loss:   {val_loss}, sum: {val_loss.sum()}')
            
            #train_loss_array[ep-1] = train_loss
            #val_loss_array[ep-1] = val_loss
            
            lr_ = optimizer.param_groups[0]['lr']
            if ep % 5 == 0:
                PATH = f'saved_models/dP_UFNOB_UNet_{UNet}_{ep}ep_{width}width_{mode1}m1_{mode2}m2_{input_array.shape[0]}train_{lr_:.2e}lr_b1_{beta1}b2_{beta2}b3_{beta3}b4_{beta4}b5_{beta5}b6_{beta6}b7_{beta7}'
                torch.save(model, PATH)
            # Framework agnostic / custom metrics
            wandb.log({"epoch": ep, "loss": train_loss.sum(), "val_loss": val_loss.sum()})
        
        
    if ufno_model == 'UFNOBR':
        training_loop_UFNOBR(model_current = 'model_head',epochs = 200)
        training_loop_UFNOBR(model_current = 'model',epochs = epochs)
    elif ufno_model == 'UFNOB':
        training_loop_UFNOB()
    

    mask = (input_array[0,:,:,0:1,0]!=0).reshape(1,nz, nx, 1, 1).repeat(1,1,1,nt,4)
    mse_function = torch.nn.MSELoss()

    def r2_function(y_pred, y):
        target_mean = torch.mean(y)
        ss_tot = torch.sum((y - target_mean) ** 2)
        ss_res = torch.sum((y - y_pred) ** 2)
        r2 = 1 - ss_res / ss_tot
        return r2
    
    def measure_metric(data_loader, sample_size, batch_size, metric='mse'):
        result = np.zeros(sample_size)
        i = 0
        for x, y in data_loader:
            nt = x.shape[-2]
            x, y = x.to(device), y.to(device)
            if ufno_model == 'UFNOB':
                y_pred = model(x.float())

            elif ufno_model == 'UFNOBR': # inference with BR model
                im_mean = torch.mean(scaled_output_4[:,:,:,0,:],axis = 0).to(device)
                im = im_mean.to(device).reshape(-1,nz,nx,4).repeat(x.shape[0],1,1,1) 
                for t in np.arange(nt): 
                    x_ = torch.cat((x[:,:,:,t,:], im), dim=-1).to(device)
                    if t == 0:
                        im = model_head(x_.float())
                    else:
                        im = model(x_.float())
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)
                y_pred = pred.reshape(x.shape[0],nz,nx,nt,4)
                if x.shape[0]==1:
                    y_pred = y_pred[0,:,:,:,:]

            if(batch_size==1):
                if(metric=='mse'):
                    result[i] = mse_function(y_pred[mask[0,...]], y[0,...][mask[0,...]])
                if(metric=='mre'):
                    result[i] = (torch.norm(y[0,...][mask[0,...]]-y_pred[mask[0,...]], 2)/torch.norm(y[0,...][mask[0,...]],2)).cpu().detach().numpy()
                if(metric=='r2'):
                    result[i] = r2_function(y_pred[mask[0,...]], y[0,...][mask[0,...]])
                i = i+1
            else:
                for b in range(batch_size):
                    try:
                        if(metric=='mse'):
                            result[i+b] = mse_function(y_pred[b,...][mask[0,...]], y[b,...][mask[0,...]])
                        if(metric=='mre'):
                            result[i+b] = (torch.norm(y[b,...][mask[0,...]]-y_pred[b,...][mask[0,...]], 2)/torch.norm(y[b,...][mask[0,...]],2)).cpu().detach().numpy()
                        if(metric=='r2'):
                            result[i+b] = r2_function(y_pred[b,...][mask[0,...]], y[b,...][mask[0,...]])
                    except:
                        pass
                i = i+batch_size
        return result

    
    
    mode_settings = {
        'Train':{
            'data': train_loader,
            'sample_size': dataset_sizes[0],
            'batch_size': batch_size
        },
        'Validation':{
            'data': val_loader,
            'sample_size': dataset_sizes[1],
            'batch_size': batch_size
        },
        'Test':{
            'data': test_loader,
            'sample_size': dataset_sizes[2],
            'batch_size': 1
        }
    }

    import os
    if(os.path.exists('./evaluations_UFNO.json')==False):
        f = open('evaluations_UFNO.json', 'w')

    import json
    # OPEN PREVIOUS RESULTS
    with open('evaluations_UFNO.json') as json_file:
        eval_file = json.load(json_file)

    # APPEND NEW RESULTS
    eval_file[str(wandb.config.as_dict())] = {}
    for mode in mode_settings.keys():
        loader = mode_settings[mode]['data']
        sample_size = mode_settings[mode]['sample_size']
        batch_size = mode_settings[mode]['batch_size']

        mre = measure_metric(data_loader=loader, sample_size=sample_size, batch_size=batch_size, metric='mre').mean()
        mse = measure_metric(data_loader=loader, sample_size=sample_size, batch_size=batch_size, metric='mse').mean()
        r2 = measure_metric(data_loader=loader, sample_size=sample_size, batch_size=batch_size, metric='r2').mean()

        # print(mre, mse, r2)
        eval_file[str(wandb.config.as_dict())][mode] = {
            'MRE': mre,
            'MSE': mse,
            'R^2': r2
        }

        print(mode + ": \nMRE: " + str(mre) + "\nMSE: " + str(mse) + "\nR2: " + str(r2) + "\n")

    # SAVE NEW RESULTS TO FILE
    file = json.dumps(eval_file)
    f = open("evaluations_UFNO.json","w")
    f.write(file)
    f.close()