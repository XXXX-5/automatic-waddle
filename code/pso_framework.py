import multiprocessing as mp
from itertools import product
import os
from surrogate_model import eval_surrogate
from datetime import datetime
import pprint
from kitsune import *
from pso import *
from parse_with_kitsune import *
logging.getLogger('pyswarms').setLevel(logging.WARNING)


def run_one(configs):
    """
    runs a single LiuerMihou attack.
    Args:
        configs (dict): generation parameters.
    Returns:
        report: summary statistics of the attack.
    """

    log_file = open(configs["log_file"], "w")

    log_file.write(pprint.pformat(configs) + "\n")

    report_file = open(configs["report_file"], "a")
    report_file.write(
        "decision \t vectorization \t search algorithm \t iter \t pkt_seen \t n_mal \t n_craft \t mal+craft \t reduction \t pos_mal \t pos_craft \t pos_ignore\n")

    print(pprint.pformat(configs))
    #
    report = craft_adversary(configs["malicious_file"], configs["init_file"], configs["adv_pcap_file"],
                             configs["mal_pcap_out"], configs["decision_type"], configs["threshold"], meta_path=configs["meta_path"],
                             model_path=configs["model_path"], optimizer=configs["optimizer"], init_count=configs["init_file_len"],
                             mutate_prob=configs["mutate_prob"], netstat_path=configs[
                                 "netstat_path"], base_offset=configs["base_offset"],
                             log_file=log_file, n_dims=configs["n_dims"], max_time_window=configs[
        "max_time_window"], max_adv_pkt=configs["max_adv_pkt"], use_seed=configs["use_seed"],
        max_craft_pkt=configs["max_craft_pkt"], max_pkt_size=configs["max_pkt_size"], adv_csv_file=configs["adv_csv_file"],
        animation_folder=configs["animation_folder"], iteration=configs["iter"], netstat_log_file=configs["netstat_log_file"])

    # evaluate on real
    pos_mal, pos_craft, pos_ignore = eval_kitsune(configs["adv_csv_file"], configs["eval_model_path"], threshold=configs["eval_threshold"], meta_file=configs["meta_path"],
                                                  ignore_index=0, out_image=configs["kitsune_graph_path"], record_scores=True)

    # eval with surrogate model
    if configs["decision_type"] == "autoencoder":
        eval_surrogate(configs["adv_csv_file"], configs["model_path"], threshold=configs["threshold"], meta_file=configs["meta_path"],
                       ignore_index=0, out_image=configs["autoencoder_graph_path"], record_scores=True)

    report["pos_mal"] = pos_mal

    report["pos_ignore"] = pos_ignore

    if report["num_altered"] == 0:
        log_file.write(pprint.pformat(report))
    else:

        if configs["mutate_prob"] == 0.5:
            alg = "PSO+DE"
        elif configs["mutate_prob"] == 1:
            alg = "DE"
        elif configs["mutate_prob"] == -1:
            alg = "PSO"

        if configs["use_seed"] == True:
            vec = configs["n_dims"] + 0.5
        else:
            vec = configs["n_dims"]
        report_file.write("\t".join(list(map(str, [configs["decision_type"], vec, alg, configs["iter"] + 1, report["num_seen"],
                                                   report["num_altered"], report["total_craft"], report["num_altered"] + report["total_craft"], report["average_reduction_ratio"], pos_mal, pos_craft, pos_ignore]))))
        report_file.write("\n")
        log_file.write(pprint.pformat(report))
    log_file.close()
    pprint.pprint(report)

    return report


def iterative_gen(max_iter, attack_configs, min_iter=0):
    """
    runs a batch of LiuerMihou attacks, parameter values should be self explainatory
    Args:
        max_iter (int): maximum iteration of an attack (when there are still packets with high anomaly score).
        optimizer (tuple): which search algorithm to use.
        decision_type (string): whether to use surrogate or kitsune as detection model.
        n_dims (tuple): specifies the vectorization method.
        attack_configs (dict): parameters for attack.
        min_iter (int): minimum iterations to start, mainly used to continous previous unfinished experiments. Defaults to 0.
    Returns:
        None
    """
    configs = attack_configs

    # configs["threshold"]=0.5445

    # configs["init_file_len"]=81838
    # configs["init_file_len"] = 14400

    # folder structure: experiment/traffic_shaping/{dataset}/["craft", "adv", "csv", "png", "anim", "meta","logs"]/{dt_t_c_d_o_m}
    base_folder = "../experiment/traffic_shaping/{}".format(
        attack_configs["name"])
    experiment_folder = "{}_{}_{}_{}_{}_{}{}".format(
        configs["decision_type"], configs["max_time_window"], configs["max_craft_pkt"], configs["n_dims"], configs["use_seed"], configs["optimizer"], configs["mutate_prob"])

    for i in ["craft", "adv", "csv", "png", "anim", "meta", "logs"]:
        if not os.path.exists(os.path.join(base_folder, i, experiment_folder)):
            os.makedirs(os.path.join(base_folder, i, experiment_folder))

    for i in range(min_iter, max_iter):
        print("iteration:", i)
        # mal_pcap file will be the next malicious_file
        configs["mal_pcap_out"] = base_folder + \
            "/craft/{}/{}_iter_{}.pcap".format(experiment_folder,
                                               configs["name"], i + 1)
        configs["adv_pcap_file"] = base_folder + \
            "/adv/{}/{}_iter_{}.pcap".format(experiment_folder,
                                             configs["name"], i)
        configs["adv_csv_file"] = base_folder + \
            "/csv/{}/{}_iter_{}.csv".format(experiment_folder,
                                            configs["name"], i)

        configs["animation_folder"] = base_folder + \
            "/anim/{}/{}_iter_{}".format(experiment_folder, configs["name"], i)
        configs["meta_path"] = base_folder + \
            "/meta/{}/{}_iter_{}.csv".format(experiment_folder,
                                             configs["name"], i)
        configs["log_file"] = base_folder + \
            "/logs/{}/{}_iter_{}.txt".format(experiment_folder,
                                             configs["name"], i)
        configs["netstat_log_file"] = base_folder + \
            "/logs/{}/netstat_{}_iter_{}.txt".format(experiment_folder,
                                                     configs["name"], i)
        configs["report_file"] = base_folder + "/logs/report.csv"
        configs["iter"] = i
        configs["kitsune_graph_path"] = base_folder + \
            "/png/{}/{}_iter{}_kitsune_rmse.png".format(
                experiment_folder, configs["name"], i)
        configs["autoencoder_graph_path"] = base_folder + \
            "/png/{}/{}_iter{}_ae_rmse.png".format(
                experiment_folder, configs["name"], i)

        # first iteration uses original malicious file, and limit packets to first 10
        if i == 0:
            # configs["malicious_file"]="../kitsune_dataset/wiretap_malicious_hostonly.pcapng"
            configs["max_adv_pkt"] = 1000

            # base offset is the time between last normal packet and first malicious packet
            # configs["base_offset"] =-596.31862402
        else:
            configs["malicious_file"] = base_folder + \
                "/craft/{}/iter_{}.pcap".format(experiment_folder, i)
            configs["max_adv_pkt"] = 1000
            configs["base_offset"] = 0

        report = run_one(configs)

        #
        if report["num_altered"] == 0 or report["craft_failed"] + report["num_failed"] == 0:
            break
