import argparse

def contains_model(path_to_model):
    pass

def package(path_to_model):
    pass

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Package model.")
    parser.add_argument("model_dir",help="Path to model directory")  
    args = parser.parse_args()
    try:
        package(args.model_dir)
    except FileNotFoundError :
        print("The given directory doesn't contain a model")