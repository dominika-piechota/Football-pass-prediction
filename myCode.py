import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from xgboost import XGBRanker
from toy_example import load_from_csv, write_submission

TRAIN_INPUT = "code/input_train_set.csv"
TRAIN_OUTPUT = "code/output_train_set.csv"
TEST_INPUT = "code/input_test_set.csv"

def softmax_with_temperature(x, temp=1.0):
    e_x = np.exp((x - np.max(x, axis=1, keepdims=True)) / temp)
    return e_x / e_x.sum(axis=1, keepdims=True)

# LightGBM MODEL
class LGBMPipeline:

    def __init__(self):

        self.ELLIPSE_RATIO = 0.85
        self.MAX_ELLIPSE_WIDTH_CM = 1500.0

        self.CANDIDATES = 22 # number of players in the field
        self.best_temp = 0.65 # softmax feature
        
        self.params = {
            "task": "train",
            "objective": "lambdarank",
            "metric": "ndcg",
            "boosting": "gbdt",
            "ndcg_eval_at": [1],
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 3,
            "lambda_l1": 0.0,
            "lambda_l2": 0.0,
            "num_leaves": 64,
            "max_depth": -1,
            "min_data_in_leaf": 20,
            "learning_rate": 0.02,
            "verbose": -1,
            "num_threads": 4
        }
        
        self.features_list = [
            "receiver_closest_opponent_dist", "norm_receiver_sender_x_diff", "receiver_closest_opp_to_sender_dist", "abs_y_diff",
            "receiver_to_sender_dist", "sender_closest_opponent_dist","receiver_closest_3_opponents_dist", "receiver_to_center_distance",
            "receiver_closest_3_teammates_dist", "min_pass_angle", "sender_dist_to_center", "dx", "dy", "same_team", "sender_zone",
            "receiver_zone", "same_pitch_zone", "pressure_diff", "local_overload", "opponents_in_elipse"
        ]

    #overwriten method from toy_example
    def make_pairs(self, X_LS, y_LS=None):

        all_rows = []
        y_rows = []
        cols = ["sender", "player_j", "x_sender", "y_sender", "x_j", "y_j", "same_team", "direction"]

        for p in range(1, 23):
            cols.append(f"x_{p}")
            cols.append(f"y_{p}")

        num_of_obs = X_LS.shape[0]
        
        for i in range(num_of_obs):

            obs = X_LS.iloc[i]
            sender_id = int(obs["sender_id"])

            # forward or backward pass? next computations later depend on this
            if sender_id <= 11:
                attack_direction = 1
            else:
                attack_direction = -1

            if y_LS is not None:
                true_reciver = int(y_LS.iloc[i].iloc[0])
            else:
                true_reciver = -1

            for potential_reciver_id in range(1, 23):
                
                # are they in the same team?
                if (sender_id <= 11 and potential_reciver_id <= 11) or (sender_id > 11 and potential_reciver_id > 11):
                    same_team = 1
                else:
                    same_team = 0
                
                potential_pass_direction = 0

                if potential_reciver_id != sender_id:
                    sender_x_coord = obs[f"x_{sender_id}"]
                    sender_y_coord = obs[f"y_{sender_id}"]
                    receiver_x_coord = obs[f"x_{potential_reciver_id}"]
                    receiver_y_coord = obs[f"y_{potential_reciver_id}"]

                    # forward or backward pass?
                    x_distance = receiver_x_coord - sender_x_coord
                    if x_distance * attack_direction > 0:
                        potential_pass_direction = 1
                    else:
                        potential_pass_direction = 0

                row = [sender_id, potential_reciver_id, sender_x_coord, sender_y_coord, receiver_x_coord, receiver_y_coord, same_team, potential_pass_direction]
                
                # all players positions
                for p in range(1, 23): 
                    row.append(obs[f"x_{p}"])
                    row.append(obs[f"y_{p}"])
                all_rows.append(row)

                if y_LS is not None: 
                    y_rows.append(int(potential_reciver_id == true_reciver))

        X_pairs = pd.DataFrame(all_rows, columns=cols)

        if y_LS is not None:
            y_pairs = pd.DataFrame(y_rows, columns=["pass"])
        else:
            y_pairs = None

        return X_pairs, y_pairs

    def lgbm_compute_features(self, df):

        output = df.copy()
        num_of_rows = output.shape[0]

        senders_x_coords = output["x_sender"].values
        senders_y_corrds = output["y_sender"].values
        potent_recivers_x_coords = output["x_j"].values
        potent_recivers_y_coords = output["y_j"].values

        all_players_x = output[[f"x_{p}" for p in range(1, 23)]].values
        all_players_y = output[[f"y_{p}" for p in range(1, 23)]].values
        
        output["dx"] = potent_recivers_x_coords - senders_x_coords
        output["dy"] = potent_recivers_y_coords - senders_y_corrds
        output["receiver_to_sender_dist"] = np.sqrt(output["dx"]**2 + output["dy"]**2)
        output["abs_y_diff"] = np.abs(potent_recivers_y_coords - senders_y_corrds)
        output["sender_dist_to_center"] = np.sqrt(senders_x_coords**2 + senders_y_corrds**2)
        output["receiver_to_center_distance"] = np.sqrt(potent_recivers_x_coords**2 + potent_recivers_y_coords**2)
        output["norm_receiver_sender_x_diff"] = output["dx"] / 10500.0

        reciver_to_closest_opp_dist = np.zeros(num_of_rows)
        reciver_to_three_cl_opp_mean = np.zeros(num_of_rows) # mean distance between receiver and 3 closests opponents to him
        reciver_to_three_cl_friends_mean = np.zeros(num_of_rows) # mean distance between receiver and 3 closests teammates to him
        sender_to_cl_opp_dist = np.zeros(num_of_rows) # distance between sender and closest opponent to him
        recivers_cl_opp_to_sender_dist = np.zeros(num_of_rows) # distance between sender and closest opponent to the receiver
        local_overload = np.zeros(num_of_rows) # number of teammates - number of opponents (in area around receiver)
        num_of_opponents_in_elipse = np.zeros(num_of_rows) # number of opponents in the elipse containing sender and a potential receiver (and some more space)
        
        # spliting the field into 9 zones
        def detect_field_zone(x, y):
             zone_x = np.where(x < -1750, 0, np.where(x < 1750, 1, 2)) # 5250 / 3 = 1750
             zone_y = np.where(y < -1133, 0, np.where(y < 1133, 1, 2))
             zone = zone_x * 3 + zone_y # unique zone id
             return zone
        
        output["sender_zone"] = detect_field_zone(senders_x_coords, senders_y_corrds)
        output["receiver_zone"] = detect_field_zone(potent_recivers_x_coords, potent_recivers_y_coords)
        output["same_pitch_zone"] = (output["sender_zone"] == output["receiver_zone"]).astype(int)

        print("geometric features computation")
        for i in range(num_of_rows):

            senders_id = int(output.iloc[i]["sender"])
            potential_recivers_id = int(output.iloc[i]["player_j"])
            
            if senders_id <= 11: 
                opponents_ids = range(11, 22) 
            else: 
                opponents_ids = range(0, 11)
            
            curr_all_players_x = all_players_x[i]
            curr_all_players_y = all_players_y[i]
            
            reciver_to_opponent_dist = np.sqrt((curr_all_players_x[opponents_ids] - potent_recivers_x_coords[i])**2 + (curr_all_players_y[opponents_ids] - potent_recivers_y_coords[i])**2)
            reciver_to_closest_opp_dist[i] = reciver_to_opponent_dist.min()
            reciver_to_three_cl_opp_mean[i] = np.sort(reciver_to_opponent_dist)[:3].mean()

            sender_to_opponent_dist = np.sqrt((curr_all_players_x[opponents_ids] - senders_x_coords[i])**2 + (curr_all_players_y[opponents_ids] - senders_y_corrds[i])**2)
            sender_to_cl_opp_dist[i] = sender_to_opponent_dist.min()
            closest_to_receiver_opponent_id = opponents_ids[np.argmin(reciver_to_opponent_dist)]
            recivers_cl_opp_to_sender_dist[i] = np.sqrt((curr_all_players_x[closest_to_receiver_opponent_id] - senders_x_coords[i])**2 + (curr_all_players_y[closest_to_receiver_opponent_id] - senders_y_corrds[i])**2)

            teammates_ids = range(0, 11) if senders_id <= 11 else range(11, 22)
            teammates_distance = []
            for t in teammates_ids:
                if t+1 != senders_id and t+1 != potential_recivers_id:
                    teammates_distance.append(np.sqrt((curr_all_players_x[t] - potent_recivers_x_coords[i])**2 + (curr_all_players_y[t] - potent_recivers_y_coords[i])**2))
            
            teammates_dist_sorted = np.sort(teammates_distance)
            three_nearest_teammates = teammates_dist_sorted[:3]
            reciver_to_three_cl_friends_mean[i] = three_nearest_teammates.mean()

            teammates_in_radius = 1
            for d in teammates_distance:
                if d<600:
                    teammates_in_radius += 1

        opponents_in_radius = 0
        for d in reciver_to_opponent_dist:
            if d<600:
                opponents_in_radius += 1
        
            local_overload[i] = teammates_in_radius - opponents_in_radius

            receiver_to_sender_dist = output.iloc[i]["receiver_to_sender_dist"]
            if receiver_to_sender_dist>10:
                sender_receiver_x_dist = potent_recivers_x_coords[i] - senders_x_coords[i]
                sender_receiver_y_dist = potent_recivers_y_coords[i] - senders_y_corrds[i]

                # elipse definition formulas
                elipse_center_x = senders_x_coords[i] + self.ELLIPSE_RATIO * sender_receiver_x_dist
                elipse_center_y = senders_y_corrds[i] + self.ELLIPSE_RATIO * sender_receiver_y_dist

                # a = elipse center to sender distance
                a_squared = (elipse_center_x - senders_x_coords[i])**2 + (elipse_center_y - senders_y_corrds[i])**2
                
                # c = elipse center to receiver distance
                c_squared = (elipse_center_x - potent_recivers_x_coords[i])**2 + (elipse_center_y - potent_recivers_y_coords[i])**2
                
                b_squared = a_squared - c_squared
                b_sq_limit = 1500.0**2
                
                if b_squared>0:
                    b_sq = min(b_squared, b_sq_limit)
                    
                    if a_squared > 0 and b_sq > 0:
                        # rotating elipse to align with sender-receiver line
                        rotation_angle = np.arctan2(sender_receiver_y_dist, sender_receiver_x_dist)
                        cos_a, sin_a = np.cos(-rotation_angle), np.sin(-rotation_angle)
                        
                        cnt = 0
                        for opponent_id in opponents_ids:
                            # transfering to good position
                            t_x, t_y = curr_all_players_x[opponent_id] - elipse_center_x, curr_all_players_y[opponent_id] - elipse_center_y
                            # rotation
                            x_rotation = t_x*cos_a - t_y*sin_a
                            y_rotation = t_x*sin_a + t_y*cos_a
                            
                            # chceck if in elipse
                            if (x_rotation**2)/a_squared + (y_rotation**2)/b_sq <= 1:
                                cnt += 1
                        num_of_opponents_in_elipse[i] = cnt

        output["receiver_closest_opponent_dist"] = reciver_to_closest_opp_dist
        output["receiver_closest_3_opponents_dist"] = reciver_to_three_cl_opp_mean
        output["receiver_closest_3_teammates_dist"] = reciver_to_three_cl_friends_mean
        output["sender_closest_opponent_dist"] = sender_to_cl_opp_dist
        output["receiver_closest_opp_to_sender_dist"] = recivers_cl_opp_to_sender_dist
        output["pressure_diff"] = reciver_to_closest_opp_dist - sender_to_cl_opp_dist
        output["local_overload"] = local_overload
        output["opponents_in_elipse"] = num_of_opponents_in_elipse
        output["min_pass_angle"] = 0

        return output[self.features_list]

    def lgbm_train_predict(self, X_train_raw, y_train_raw, X_test_raw):

        print("Preparing Training Data")
        X_pairs, y_pairs = self.make_pairs(X_train_raw, y_train_raw)
        X_features = self.lgbm_compute_features(X_pairs)
        y_values = y_pairs["pass"].values
        
        num_of_queries = len(y_values) // self.CANDIDATES
        groups = [self.CANDIDATES] * num_of_queries
        
        print(f"LGBM training on {len(y_values)} rows ({num_of_queries} queries)")
        dataset_train = lgb.Dataset(X_features, label=y_values, group=groups)
        lgbm_model = lgb.train(self.params, dataset_train, num_boost_round=800)
        
        print("LGBM predicting Test")
        X_test_pairs, _ = self.make_pairs(X_test_raw, None)
        X_test_features = self.lgbm_compute_features(X_test_pairs)

        raw_preds = lgbm_model.predict(X_test_features)
        raw_preds = raw_preds.reshape(-1, self.CANDIDATES)
        probas = softmax_with_temperature(raw_preds, temp=self.best_temp)

        return probas, X_test_pairs

class XGBPipeline:
    def __init__(self):
        self.RADIUS = 500.0 # cm
        self.ELLIPSE_RATIO = 0.85
        self.CANDIDATES = 22
        self.features_list = [
            "dx", "dy", "total_distance", "receiver_opponent_dist", "average_three_receiver_opponents", 
            "average_three_receiver_friendly", "distance_sender_closest_opponent", "angle", "direction", 
            "sender_zone_x", "sender_zone_y", "receiver_zone_x", "receiver_zone_y", "zone_progression", "opponents_in_elipse"
        ]

    def xgb_precompute_stats(self, x_raw):

        num_of_obs = x_raw.shape[0]

        dist_to_cl_opp_min = np.zeros((num_of_obs, 22)) # player distance to the closest opponent
        dist_to_three_opp_avg = np.zeros((num_of_obs, 22)) # average of the distances to the three closest opponents for each player
        dist_to_three_friends_avg = np.zeros((num_of_obs, 22)) # average of the distances to the three closest teammates for each player
        
        print(f"XGB precomputing stats")
        for i in range(num_of_obs):

            curr_row = x_raw.iloc[i]

            x_coords = np.array([curr_row[f"x_{p}"] for p in range(1, 23)])
            y_coords = np.array([curr_row[f"y_{p}"] for p in range(1, 23)])

            for p in range(22):
                curr_x, curr_y = x_coords[p], y_coords[p]
                distances = np.sqrt((x_coords - curr_x)**2 + (y_coords - curr_y)**2)
                distances[p] = np.inf # setting distance to self as infinite because we dont want the model to predict self-passes

                zero_to_eleven_team = 0 # team falg
                if p<11:
                    zero_to_eleven_team = 0
                else:
                    zero_to_eleven_team = 1

                opponent_mask = np.array([(k < 11) != (zero_to_eleven_team == 0) for k in range(22)])
                friend_mask   = ~opponent_mask

                opponents_distances = distances[opponent_mask]
                firiends_distances  = distances[friend_mask]

                dist_to_cl_opp_min[i, p] = opponents_distances.min()
                dist_to_three_opp_avg[i, p] = np.mean(np.sort(opponents_distances)[:3])
                dist_to_three_friends_avg[i, p]  = np.mean(np.sort(firiends_distances)[:3])

        return {"min_dist_opp": dist_to_cl_opp_min, "avg3_opp": dist_to_three_opp_avg, "avg3_fr": dist_to_three_friends_avg}

    def xgb_build_features(self, x_raw, stats, y_raw=None):

        rows = []
        rows_y = []
        
        stat_dist_to_opp_min = stats["min_dist_opp"]
        stat_dist_to_three_opp_avg = stats["avg3_opp"]
        stat_dist_to_three_friends_avg = stats["avg3_fr"]

        num_of_samples = x_raw.shape[0]
        
        for i in range(num_of_samples):

            curr_obs = x_raw.iloc[i]
            sender_id = int(curr_obs["sender_id"])

            if y_raw is not None:
                true_reciver = int(y_raw.iloc[i].iloc[0])
            else:
                true_reciver = -1

            sender_x = curr_obs[f"x_{sender_id}"]
            sender_y = curr_obs[f"y_{sender_id}"]

            # adding features for all potential receivers
            for pot_rec_id in range(1, 23):

                pot_reciver_x = curr_obs[f"x_{pot_rec_id}"]
                pot_reciver_y = curr_obs[f"y_{pot_rec_id}"]

                sender_idx = sender_id - 1
                receiver_idx = pot_rec_id - 1

                dx = pot_reciver_x - sender_x
                dy = pot_reciver_y - sender_y
                sender_to_pot_reiver_dist = np.sqrt(dx**2 + dy**2)
                
                sender_zone_x = 1 if sender_x < -1750 else (2 if sender_x < 1750 else 3)
                sender_zone_y = 1 if sender_y < -1133 else (2 if sender_y < 1133 else 3)
                pot_receiver_zone_x = 1 if pot_reciver_x < -1750 else (2 if pot_reciver_x < 1750 else 3)
                pot_receiver_zone_y = 1 if pot_reciver_y < -1133 else (2 if pot_reciver_y < 1133 else 3)

                features = {
                    "dx": dx, "dy": dy,
                    "total_distance": sender_to_pot_reiver_dist,
                    "receiver_opponent_dist": stat_dist_to_opp_min[i, receiver_idx],
                    "average_three_receiver_opponents": stat_dist_to_three_opp_avg[i, receiver_idx],
                    "average_three_receiver_friendly": stat_dist_to_three_friends_avg[i, receiver_idx],
                    "distance_sender_closest_opponent": stat_dist_to_opp_min[i, sender_idx],
                    "angle": 0,
                    "direction": -0.7* abs(dx) if dx > 0 else 0.2*abs(dx), # the smaller the direction value the better (straight forward passes less penalized)
                    "sender_zone_x": sender_zone_x, 
                    "sender_zone_y": sender_zone_y,
                    "receiver_zone_x": pot_receiver_zone_x, 
                    "receiver_zone_y": pot_receiver_zone_y,
                    "zone_progression": pot_receiver_zone_x - sender_zone_x, 
                    "opponents_in_elipse": 0
                }
                rows.append(features)

                if y_raw is not None: 
                    rows_y.append(int(pot_rec_id == true_reciver))
                
        return pd.DataFrame(rows)[self.features_list], np.array(rows_y) if y_raw is not None else None

    def xgb_train_predict(self, X_train_raw, y_train_raw, X_test_raw):

        print("XGB precomputing stats")
        stats_train = self.xgb_precompute_stats(X_train_raw)
        stats_test = self.xgb_precompute_stats(X_test_raw)
        
        print("XGB building features")
        X_feat, y_vals = self.xgb_build_features(X_train_raw, stats_train, y_train_raw)
        
        print("XGB training")
        xgb_model = XGBRanker(
            objective='rank:ndcg',
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=7,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            random_state=42,
            eval_metric="ndcg@1"
        )
        groups = [self.CANDIDATES] * (len(y_vals) // self.CANDIDATES)
        xgb_model.fit(X_feat, y_vals, group=groups, verbose=False)
        
        print("XGB predicting on test set")
        X_test_feat, _ = self.xgb_build_features(X_test_raw, stats_test, None)
        xgb_scores = xgb_model.predict(X_test_feat)
        xgb_scores = xgb_scores.reshape(-1, self.CANDIDATES)
        probas = softmax_with_temperature(xgb_scores, temp=1.0)
        return probas
    
def main():

    if not os.path.exists(TRAIN_INPUT):
        print(f"Input file not found.")
        return

    print("LOADING DATA")
    X_LS = load_from_csv(TRAIN_INPUT)
    y_LS = load_from_csv(TRAIN_OUTPUT)
    X_TS = load_from_csv(TEST_INPUT)

    print("LIGHTGBM PIPELINE")
    lgbm_pipe = LGBMPipeline()
    probas_lgbm, _ = lgbm_pipe.lgbm_train_predict(X_LS, y_LS, X_TS)
    
    print("XGBOOST PIPELINE")
    xgb_pipe = XGBPipeline()
    probas_xgb = xgb_pipe.xgb_train_predict(X_LS, y_LS, X_TS)

    print("MIXING MODELS")
    final_probabilities = 0.5 * probas_lgbm + 0.5 * probas_xgb
    
    output_file = "submission_ensemble_fixed.csv"
    write_submission(
        probas=final_probabilities,
        estimated_score=0.4120,
        file_name=output_file
    )
    print(f"\n\n Saved to {output_file}")

if __name__ == "__main__":
    main()