import json

class Asset:

    def __init__(self,ticker:str,shares:float, avg_cost:float):
        self.ticker= ticker
        self.shares=shares
        self.avg_cost=avg_cost
    
    def __repr__(self):
        return f"['{self.ticker}', {self.shares}, {self.avg_cost}]"
    
    def calculate_market_value(self,current_price):
        '''
        calculates the value in the market
        '''
        market_value=current_price*self.shares
        return market_value

    def calculate_pnl(self,current_price):
        '''
        Calculates the PnL (Profit and Loss)
        '''
        return (current_price - self.avg_cost) * self.shares
    
    def to_dict(self):
        data={}
        data['ticker']= self.ticker
        data['shares']=self.shares
        data['avg_cost']=self.avg_cost

        return data

class Portfolio:
    def __init__(self):
        self.assets=[]
        self.balance=0.0
        
    def add_asset(self,asset:object):
        self.assets.append(asset)
    
    def get_total_value(self,market_data:dict):
        total=0
        for asset in self.assets:
            value=market_data[asset.ticker]
            sub_total=asset.calculate_market_value(value)
            total+=sub_total
        return total

    def get_portfolio_pnl(self,market_data:dict):
        total=0
        for asset in self.assets:
            value=market_data[asset.ticker]
            sub_total=asset.calculate_pnl(value)
            total+=sub_total
        return total                

    def sell_asset(self,ticker:str,shares_sold:float, sell_price:float):
        for asset in self.assets:
            if asset.ticker==ticker:
                if asset.shares >= shares_sold:
                    pnl= (sell_price - asset.avg_cost) * shares_sold
                    asset.shares=asset.shares - shares_sold
                    self.balance += shares_sold*sell_price
                    return pnl
                elif asset.shares < shares_sold:
                    return'Insufficient funds'
                
        return 'The ticket does not match'
    
    def save_portafolio(self, filename):
        data_list=[]
        for asset in self.assets:
            data_list.append(asset.to_dict())
        
        datos_a_guardar = {
                "assets": data_list,
                "balance": self.balance
                        }
        with open(filename,'w') as variable:
            json.dump(datos_a_guardar,variable)
    
    def load_portafolio(self,filename):
        with open(filename,'r') as archivo:
            datos=json.load(archivo)
        
        self.balance=datos['balance']

        self.assets= []

        for item in datos['assets']:
            new_asset= Asset(item['ticker'], item['shares'],item['avg_cost'])

            self.assets.append(new_asset)
        
