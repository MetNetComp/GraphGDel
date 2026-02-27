import torch

def save_model_with_confirmation(model, CBM, Input_seq):
    """
    Save the model only after user confirmation.

    Parameters:
    model (torch.nn.Module): The PyTorch model to be saved.
    CBM (str): The CBM identifier for naming the file.
    """
    model_path = f'DeepGdel_DecoderV2_hp_{CBM}_temp.sav'
    
    # Ask for user confirmation
    confirm = input("Enter '1' to confirm saving the model: ")
    
    if confirm == '1':
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}!")
    else:
        print("Model saving canceled.")