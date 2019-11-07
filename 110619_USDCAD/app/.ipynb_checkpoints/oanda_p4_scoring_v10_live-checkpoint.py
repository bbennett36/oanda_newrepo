import pandas as pd
import requests
import json
import requests
import time as t
from datetime import datetime, timedelta
from IPython.display import display, clear_output
import numpy as np
import lightgbm as lgb
import argparse
from IPython.display import display, clear_output

def get_candle_data(pair, count):
    count = str(count)
    num_cols = ['volume', 'bid_o', 'bid_h','bid_l','bid_c', 'ask_o', 'ask_h','ask_l','ask_c']

    endpoint = "https://api-fxtrade.oanda.com/v3/instruments/" + pair + "/candles?&count=" + count + "&price=BA&granularity=H1&alignmentTimezone=America/New_York&dailyAlignment=0"
    headers = {"Authorization": "Bearer cffb90f2341ed281be6e4910799d2e77-c1c86c039d32bae718f9e7c213784182"}
    response = requests.get(url=endpoint, headers=headers)
    data = json.loads(response.text)
    out_temp = pd.DataFrame()
    counter = 1

    for d in data['candles']:
        temp = pd.DataFrame()
        temp['instrument'] = [data['instrument']]
        temp['volume'] = d['volume']        
        temp['time'] = d['time']

        temp['bid_o'] = d['bid']['o']
        temp['bid_h'] = d['bid']['h']
        temp['bid_l'] = d['bid']['l']
        temp['bid_c'] = d['bid']['c']

        temp['ask_o'] = d['ask']['o']
        temp['ask_h'] = d['ask']['h']
        temp['ask_l'] = d['ask']['l']
        temp['ask_c'] = d['ask']['c']
        
        if counter == 1:
            out_temp = temp
            counter +=1 
        else:
            out_temp = out_temp.append(temp)


    return out_temp

def get_ob_data(price_range_iter, pair, candle_data):
    num_cols = ['bucketwidth', 'price', 'longcountpercent', 'shortcountpercent']
    try:
        tempcandle = candle_data.copy()
        endpoint = "https://api-fxtrade.oanda.com/v3/instruments/" + pair + "/orderBook"
        headers = {"Authorization": "Bearer cffb90f2341ed281be6e4910799d2e77-c1c86c039d32bae718f9e7c213784182"}
        response = requests.get(url=endpoint, headers=headers)
        data = json.loads(response.text)
        temp = pd.DataFrame()
        temp['instrument'] = [data['orderBook']['instrument']]
        temp['time'] = data['orderBook']['time']
        temp['time'] = pd.to_datetime(temp['time'])
        temp['bucketwidth'] = data['orderBook']['bucketWidth']

        ask_high = round(tempcandle['ask_h'].values[0], 4)
        ask_high_max = round(ask_high + 0.015, 4)
        bid_low = round(tempcandle['bid_l'].values[0], 4)
        bid_low_min = round(bid_low - 0.015, 4)
        mid_prev_close = round((ask_high + bid_low)/2, 4)
        ask_close =  round(tempcandle['ask_c'].values[0], 4)
        bid_close =  round(tempcandle['bid_c'].values[0], 4)
    
        ob_df = pd.DataFrame()
        ob_counter = 1
        for d in data['orderBook']['buckets']:
            if float(d['price']) <= ask_high_max and float(d['price']) >= bid_low_min:
                temp_ob = pd.DataFrame()
                temp_ob['price'] = [float(d['price'])]
                temp_ob['longcountpercent'] = float(d['longCountPercent'])
                temp_ob['shortcountpercent'] = float(d['shortCountPercent'])
                if ob_counter == 1:
                    ob_df = temp_ob
                    ob_counter +=1
                else:
                    ob_df = ob_df.append(temp_ob)
        
        ob_range = np.arange(ask_close-0.015, (ask_close+0.015)+price_range_iter, price_range_iter)
        
        for v in ob_range:
#             v = round(v, 4)
#             v_col = str(v)
#             v_col = v_col.replace('0.00', '')
            price_col = str(round(ask_close-v, 4)).replace('0.00', '')
            long_price_col = 'ob_price_long_' + str(price_col).replace('.', '_') + '_csum'
            long_ratio_col = 'ob_long_ratio_' + str(price_col).replace('.', '_')
            short_price_col = 'ob_price_short_' + str(price_col).replace('.', '_') + '_csum' 
            short_ratio_col = 'ob_short_ratio_' + str(price_col).replace('.', '_')
            diff_col = 'ob_diff_' + str(price_col).replace('.', '_')
            
            if v < ask_close:  
                under_mask = (ob_df['price'] < ask_close) & (ob_df['price'] > ask_close-0.015)

                temp[long_price_col] = round(ob_df[under_mask]['longcountpercent'].cumsum().max(), 2)
                temp[short_price_col] = round(ob_df[under_mask]['shortcountpercent'].cumsum().max(), 2)
                ratio_total = round(ob_df[under_mask]['longcountpercent'].cumsum().max(), 2) + round(ob_df[under_mask]['shortcountpercent'].cumsum().max(), 2)
                temp[long_ratio_col] = round(ob_df[under_mask]['longcountpercent'].cumsum().max(), 2) / ratio_total
                temp[short_ratio_col] = round(ob_df[under_mask]['shortcountpercent'].cumsum().max(), 2) / ratio_total
                temp[diff_col] =  round(ob_df[under_mask]['longcountpercent'].cumsum().max(), 2) - round(ob_df[under_mask]['shortcountpercent'].cumsum().max(), 2)            
                
            elif v > ask_close:
                over_mask = (ob_df['price'] > ask_close) & (ob_df['price'] < ask_close+0.015) 

                temp[long_price_col] =  round(ob_df[over_mask]['longcountpercent'].cumsum().max(), 2)
                temp[short_price_col] =  round(ob_df[over_mask]['shortcountpercent'].cumsum().max(), 2)
                ratio_total = round(ob_df[over_mask]['longcountpercent'].cumsum().max(), 2) + round(ob_df[over_mask]['shortcountpercent'].cumsum().max(), 2)
                temp[long_ratio_col] = round(ob_df[over_mask]['longcountpercent'].cumsum().max(), 2) / ratio_total
                temp[short_ratio_col] = round(ob_df[over_mask]['shortcountpercent'].cumsum().max(), 2) / ratio_total
                temp[diff_col] =  round(ob_df[over_mask]['longcountpercent'].cumsum().max(), 2) - round(ob_df[over_mask]['shortcountpercent'].cumsum().max(), 2)

   
        return temp

    except IndexError:
#         print('i error', t, 'pos')
        pass

    except KeyError:
#         print('k error', t, 'pos')
        pass

def get_pos_data(price_range_iter, pair, candle_data):
    num_cols = ['pos_bucketwidth', 'price', 'pos_longcountpercent', 'pos_shortcountpercent']  
    try:
        tempcandle = candle_data.copy()
        endpoint = "https://api-fxtrade.oanda.com/v3/instruments/" + pair + "/positionBook"
        headers = {"Authorization": "Bearer cffb90f2341ed281be6e4910799d2e77-c1c86c039d32bae718f9e7c213784182"}
        response = requests.get(url=endpoint, headers=headers)
        data = json.loads(response.text)
        temp = pd.DataFrame()
        temp['instrument'] = [data['positionBook']['instrument']]
        temp['time'] = data['positionBook']['time']
        temp['time'] = pd.to_datetime(temp['time'])
        temp['bucketwidth'] = data['positionBook']['bucketWidth']

        ask_high = tempcandle['ask_h'].values[0]
        ask_high_max = ask_high + 0.015
        bid_low = tempcandle['bid_l'].values[0]
        bid_low_min = bid_low - 0.015
        mid_prev_close = (ask_high + bid_low)/2
        
        ask_close =  round(tempcandle['ask_c'].values[0], 4)
        bid_close =  round(tempcandle['bid_c'].values[0], 4)
        
        pos_df = pd.DataFrame()
        pos_counter = 1
        for d in data['positionBook']['buckets']:
            if float(d['price']) <= ask_high_max and float(d['price']) >= bid_low_min:
                temp_ob = pd.DataFrame()
                temp_ob['price'] = [float(d['price'])]
                temp_ob['longcountpercent'] = float(d['longCountPercent'])
                temp_ob['shortcountpercent'] = float(d['shortCountPercent'])
                if pos_counter == 1:
                    pos_df = temp_ob
                    pos_counter +=1
                else:
                    pos_df = pos_df.append(temp_ob)
                    
                    
                    
        pos_range = np.arange(ask_close-0.015, (ask_close+0.015)+price_range_iter, price_range_iter)
        
        for v in pos_range:
#             v = round(v, 4)
#             v_col = str(v)
#             v_col = v_col.replace('0.00', '')
            price_col = str(round(ask_close-v, 4)).replace('0.00', '')
            long_price_col = 'pos_price_long_' + str(price_col).replace('.', '_') + '_csum'
            long_ratio_col = 'pos_long_ratio_' + str(price_col).replace('.', '_')
            short_price_col = 'pos_price_short_' + str(price_col).replace('.', '_') + '_csum' 
            short_ratio_col = 'pos_short_ratio_' + str(price_col).replace('.', '_')
            diff_col = 'pos_diff_' + str(price_col).replace('.', '_')
            
            if v < ask_close:  
                under_mask = (pos_df['price'] < ask_close) & (pos_df['price'] > ask_close-0.015) 

                temp[long_price_col] = round(pos_df[under_mask]['longcountpercent'].cumsum().max(), 2)
                temp[short_price_col] = round(pos_df[under_mask]['shortcountpercent'].cumsum().max(), 2)
                ratio_total = round(pos_df[under_mask]['longcountpercent'].cumsum().max(), 2) + round(pos_df[under_mask]['shortcountpercent'].cumsum().max(), 2)
                temp[long_ratio_col] = round(pos_df[under_mask]['longcountpercent'].cumsum().max(), 2) / ratio_total
                temp[short_ratio_col] = round(pos_df[under_mask]['shortcountpercent'].cumsum().max(), 2) / ratio_total     
                
            elif v > ask_close:
                over_mask = (pos_df['price'] > ask_close) & (pos_df['price'] < ask_close+0.015) 

                temp[long_price_col] =  round(pos_df[over_mask]['longcountpercent'].cumsum().max(), 2)
                temp[short_price_col] =  round(pos_df[over_mask]['shortcountpercent'].cumsum().max(), 2)

                ratio_total = round(pos_df[over_mask]['longcountpercent'].cumsum().max(), 2) + round(pos_df[over_mask]['shortcountpercent'].cumsum().max(), 2)

                temp[long_ratio_col] = round(pos_df[over_mask]['longcountpercent'].cumsum().max(), 2) / ratio_total
                temp[short_ratio_col] = round(pos_df[over_mask]['shortcountpercent'].cumsum().max(), 2) / ratio_total

                temp[diff_col] =  round(pos_df[over_mask]['longcountpercent'].cumsum().max(), 2) - round(pos_df[over_mask]['shortcountpercent'].cumsum().max(), 2) 
                    
        return temp
            

    except IndexError:
        pass

    except KeyError:
        pass
        

        
def main():
    account_id = '001-001-2676381-001'
    price_range_iter = 0.0005
    pair = 'USD_CAD'

    short_model = lgb.Booster(model_file='../model/oanda_USDCAD_short.txt')
    short_model_info = pd.read_csv('../model/oanda_USDCAD_short_info.csv')
    short_target_diff = short_model_info['long_target_diff'][0]
    short_stop_loss = short_model_info['long_stop_loss'][0]
    short_model_cols = pd.read_csv('../model/oanda_USDCAD_short_layout.csv')
    short_model_cols = short_model_cols['features'].values
    short_target_cutoff = short_model_info['l_target_cutoff'].values[0]
    
    long_model = lgb.Booster(model_file='../model/oanda_USDCAD_long.txt')
    long_model_info = pd.read_csv('../model/oanda_USDCAD_long_info.csv')
    long_target_diff = long_model_info['long_target_diff'][0]
    long_stop_loss = long_model_info['long_stop_loss'][0]
    long_model_cols = pd.read_csv('../model/oanda_USDCAD_long_layout.csv')
    long_model_cols = long_model_cols['features'].values
    long_target_cutoff = long_model_info['l_target_cutoff'].values[0]    

    safe = True
    losing_streak = 0
    counter = 24
    order_id = None

    while safe:
        try:
            time_now = datetime.now()
            is_ready = False

            candle_data = get_candle_data(pair, counter)
            candle_data = candle_data.reset_index(drop=True)
            candle_data['id'] = candle_data.index

            for c in list(candle_data):
                if c != 'time' and c != 'instrument':
                    candle_data[c] = pd.to_numeric(candle_data[c])
                    
            orig_candle_data = candle_data[candle_data['id'] == candle_data['id'].max()].copy()
            orig_candle_data.drop(['id'], axis=1, inplace=True)
            orig_candle_data = orig_candle_data.reset_index(drop=True)
    
            shift_fields = ['bid_o', 'bid_h','bid_l','bid_c','ask_o','ask_h','ask_l','ask_c']

            for s in shift_fields:
                candle_data[s] = candle_data[s].shift(1)

            candle_data = candle_data[~candle_data['bid_o'].isnull()]

            shift_counter = 1
            shift_counter_max = 24
            new_cols = []
            while shift_counter <= shift_counter_max:
                for s in shift_fields:
                    col = s+'_past'+str(shift_counter)
                    new_cols.append(col)
                    candle_data[col] = ((candle_data[s] - candle_data[s].shift(shift_counter)) / candle_data[s].shift(shift_counter))*100
                shift_counter+=1

            candle_data = candle_data[candle_data['id'] == candle_data['id'].max()]
            candle_data.drop(['id'], axis=1, inplace=True)
            candle_data = candle_data.reset_index(drop=True)
            candle_data.drop(shift_fields, axis=1, inplace=True)
            candle_data = pd.concat([candle_data, orig_candle_data[shift_fields]], axis=1)
            
            ob_data = get_ob_data(price_range_iter, pair, candle_data)
            pos_data = get_pos_data(price_range_iter, pair, candle_data)

            group1 = pd.merge(ob_data, pos_data, how='outer', on='time')

            group1['time'] = pd.to_datetime(group1['time'])
            candle_data['time'] = pd.to_datetime(candle_data['time'])
            group2 = pd.concat([group1[[c for c in list(group1) if c not in candle_data]], candle_data], axis=1)
            group2 = group2.loc[:,~group2.columns.duplicated()]

            group2['time'] = pd.to_datetime(group2['time'])
            group2['hour'] = group2['time'].dt.hour
            group2['day_of_week'] = group2['time'].dt.dayofweek
            
            for f in short_model_cols:
                group2[f] = pd.to_numeric(group2[f])
            print()
            group2['short_prob'] = short_model.predict(group2[short_model_cols])
            display('short:', datetime.now(), float(group2['short_prob'].values[0]))
            group2['short_pred'] = group2['short_prob'].apply(lambda x: 1 if x >= short_target_cutoff else 0)

            group2['long_prob'] = long_model.predict(group2[long_model_cols])
            display('long:', datetime.now(), float(group2['long_prob'].values[0]))
            group2['long_pred'] = group2['long_prob'].apply(lambda x: 1 if x >= long_target_cutoff else 0)

            if group2['short_pred'].values[0] == 1 and group2['long_pred'].values[0] == 0:
                tp_price = float(group2['ask_c'].values[0]) + short_model_info['long_target_diff'].values[0]
                tp_price = round(tp_price, 4)
                sl_price = float(group2['bid_c'].values[0]) - short_model_info['long_stop_loss'].values[0]
                sl_price = round(sl_price, 4)

                order_config = {}
                order_config['order'] = {
                    "units": "-2000",
                    "instrument": pair,    
                    "timeInForce": "FOK",
                    "type": "MARKET",
                    "positionFill": "DEFAULT",
                    "takeProfitOnFill": {
                        "price": str(sl_price)
                    },
                    "stopLossOnFill": {
                        "price": str(tp_price)
                    }
                }

                endpoint = "https://api-fxtrade.oanda.com/v3/accounts/" + account_id + "/orders"
                headers = {"Authorization": "Bearer cffb90f2341ed281be6e4910799d2e77-c1c86c039d32bae718f9e7c213784182"}
                response = requests.post(url=endpoint, json=order_config, headers=headers)
                resp_dict = json.loads(response.text)
                tp_id = resp_dict['relatedTransactionIDs'][len(resp_dict['relatedTransactionIDs'])-2]
                sl_id = resp_dict['relatedTransactionIDs'][len(resp_dict['relatedTransactionIDs'])-1]
                order_closed = False
                print('order placed')
                while not order_closed:
                    try:
                        endpoint = "https://api-fxtrade.oanda.com/v3/accounts/" + account_id + "/orders/" + str(tp_id)
                        headers = {"Authorization": "Bearer cffb90f2341ed281be6e4910799d2e77-c1c86c039d32bae718f9e7c213784182"}
                        tp_response = requests.get(url=endpoint, headers=headers)
                        tp_dict = json.loads(tp_response.text)

                        endpoint = "https://api-fxtrade.oanda.com/v3/accounts/" + account_id + "/orders/" + str(sl_id)
                        headers = {"Authorization": "Bearer cffb90f2341ed281be6e4910799d2e77-c1c86c039d32bae718f9e7c213784182"}
                        sl_response = requests.get(url=endpoint, headers=headers)
                        sl_dict = json.loads(sl_response.text)

                        if tp_dict['order']['state'] == 'FILLED':
                            order_closed = True
                            losing_streak = 0
                        elif sl_dict['order']['state'] == 'FILLED':
                            order_closed = True
                            losing_streak += 1
                        else:
            #                 print('waiting for order to close')
                            t.sleep(60)
                    except Exception as ex:
                        print('error waiting for order')
                        print(ex)
                        t.sleep(60*5)
                        pass
                    
            elif group2['short_pred'].values[0] == 0 and group2['long_pred'].values[0] == 1:
                tp_price = float(group2['ask_c'].values[0]) + long_model_info['long_target_diff'].values[0]
                tp_price = round(tp_price, 4)
                sl_price = float(group2['bid_c'].values[0]) - long_model_info['long_stop_loss'].values[0]
                sl_price = round(sl_price, 4)

                order_config = {}
                order_config['order'] = {
                    "units": "2000",
                    "instrument": pair,    
                    "timeInForce": "FOK",
                    "type": "MARKET",
                    "positionFill": "DEFAULT",
                    "takeProfitOnFill": {
                        "price": str(tp_price)
                    },
                    "stopLossOnFill": {
                        "price": str(sl_price)
                    }
                }

                endpoint = "https://api-fxtrade.oanda.com/v3/accounts/" + account_id + "/orders"
                headers = {"Authorization": "Bearer cffb90f2341ed281be6e4910799d2e77-c1c86c039d32bae718f9e7c213784182"}
                response = requests.post(url=endpoint, json=order_config, headers=headers)
                resp_dict = json.loads(response.text)
                tp_id = resp_dict['relatedTransactionIDs'][len(resp_dict['relatedTransactionIDs'])-2]
                sl_id = resp_dict['relatedTransactionIDs'][len(resp_dict['relatedTransactionIDs'])-1]
                order_closed = False
                print('order placed')
                while not order_closed:
                    try:
                        endpoint = "https://api-fxtrade.oanda.com/v3/accounts/" + account_id + "/orders/" + str(tp_id)
                        headers = {"Authorization": "Bearer cffb90f2341ed281be6e4910799d2e77-c1c86c039d32bae718f9e7c213784182"}
                        tp_response = requests.get(url=endpoint, headers=headers)
                        tp_dict = json.loads(tp_response.text)

                        endpoint = "https://api-fxtrade.oanda.com/v3/accounts/" + account_id + "/orders/" + str(sl_id)
                        headers = {"Authorization": "Bearer cffb90f2341ed281be6e4910799d2e77-c1c86c039d32bae718f9e7c213784182"}
                        sl_response = requests.get(url=endpoint, headers=headers)
                        sl_dict = json.loads(sl_response.text)

                        if tp_dict['order']['state'] == 'FILLED':
                            order_closed = True
                            losing_streak = 0
                        elif sl_dict['order']['state'] == 'FILLED':
                            order_closed = True
                            losing_streak += 1
                        else:
            #                 print('waiting for order to close')
                            t.sleep(60)
                    except Exception as ex:
                        print('error waiting for order')
                        print(ex)
                        t.sleep(60*5)
                        pass

            else:
                while not is_ready:
#                     dnow = datetime.now()
#                     dnow = dnow.minute
#                     if dnow >= 50 and dnow <=60:
#                         is_ready = True
#                         t.sleep(60)
#                     else:
#                         pass
                    result = (datetime.now() - time_now).total_seconds() / 60
                    if result >= 10:
                        is_ready = True
                    else:
                        pass
        except Exception as ex:
            print('error')
            print(ex)
            pass
                
if __name__ == '__main__':
    main()