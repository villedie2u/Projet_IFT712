""" Main projet_IFT712"""
from classes.parser import Parser
from classes.csvdataframe import CSVDataFrame
import reg_log

def main():
    print("============= Main start ===============\n")

    data = Parser("data/leaf_train.csv")

    print("> les 5 premiers paramètres sont:", data.parameters[:5])
    print("> la 15ième donnée est ", data.data[14])


    print("\n============= Main end ===============")
    
    
    """ Dataframe ordonné par l'id """
    df = CSVDataFrame("data/leaf_train.csv").data.sort_values(by=['id'])
    
    name_train_attributes = df.columns[2:]
    name_target_attribute = df.columns[1]
    df_train = df.iloc[:,2:].values
    df_target = df.iloc[:,1].values
    
    """modification de df_target avec les noms de classes fusionnées (on ne garde que le préfixe)"""
    df_target_fusion = df_target
    
    for i in range(len(df_target_fusion)):
        s = df_target_fusion[i].split('_')
        df_target_fusion[i] = s[0]
        
    #print(df_target_fusion)
        
    
    #print(name_train_attributes)
    #print(name_target_attribute)
    #print(df_target)
    #print(df_train)
    
    """pour lancer la régression logistique avec les targets non fusionnées"""
    #reg_log.reg_log(df_train,df_target)
    """pour lancer la régression logistique avec les targets fusionnées """
    #reg_log.reg_log(df_train,df_target_fusion)


if __name__ == "__main__":
    main()
