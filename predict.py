import input_args_predict as args
import predict_utils 

   
#load model

if __name__=='__main__':

    
    model1=predict_utils.restored_model(args.result.checkpoint.name)
    model1=model1.return_model()
    predict_utils.predict(args.result.image.name,model1,args.result.topk)


    


