import os
import logging
import pandas as pd                       
import numpy as np
import json
from glob import glob
from fpdf import FPDF
from PyPDF2 import PdfMerger
from analytics_toolkits.mm_analyzer import *

"""
Problem: mm
Given that there can always be overlap among periods that orders are live. The way I think about meeting the requirement is that:
- step 1: Given a particular order with live period t0, the market maker satisfies the time requirement when all the orders that are placed 
within t0 (inclusive) satisfy the requirement (both spread & size requirements)
- step 2: Once collecting all valid periods from step, aggregate non-overlapping periods in O(N)
- step 3: As it is possible that market maker's trades do not cover the whole trade session (i.e., gap period between trades), I use orders to determine the overall
period of trading session (though for this exercise, the results is the same as using the time delta between start and end timestamp in the config).
"""
def mm_analyzer(mm_data, mm_config):
    logging.basicConfig(filename='mm.log', filemode='a', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S',level=logging.INFO)
    mm_new = mm_data.apply(lambda row: mm_check_requirements(row,mm_config), axis=1)
    mm_new["time_diff_seconds"] = (mm_new["completed_ts"] - mm_new["entered_ts"]) / pd.Timedelta(seconds=1)
    mm_new = mm_new.sort_values(by=["instrument","entered_ts","completed_ts"]).reset_index(drop=True)
    instruments = mm_data.instrument.unique()
    
    logging.info("mm program starts")
    msg = 'Problem: mm\n\nMarket Making Checker\n'
    for instrument in instruments:
        logging.info(f"processing {instrument}")
        required_time_ratio = mm_config[instrument]["requirement_time"]
        # uncomment below if using config to determine length of trading session
        #ttl_time = (datetime.strptime(mm_config[instrument]["end"],"%H:%M") - datetime.strptime(mm_config[instrument]["start"], "%H:%M")).seconds
        mm_unit = mm_new.query(f"instrument=='{instrument}'")

        good_time_interval = [] # for storing valid unique time intervals
        trade_time_interval = [] # for storing all orders' unique time intervals
        unique_time_interval = set()

        for row in mm_unit.iterrows():
            start_ts, end_ts = row[1]["entered_ts"], row[1]["completed_ts"]
            item = ((start_ts, end_ts),row[1]["time_diff_seconds"])
            orders_in_interval = mm_unit[(mm_unit["entered_ts"]>=start_ts) & (mm_unit["completed_ts"]<=end_ts)]
            if (start_ts, end_ts) not in unique_time_interval:
                trade_time_interval.append(item)
                if (orders_in_interval["size_spread_requirement_met"]=='Y').all():
                    good_time_interval.append(item)
            unique_time_interval.add((start_ts,end_ts))
        
        # compute valid time and total trade time
        ttl_good_time = compute_ttl_time_meet_requirement(good_time_interval)
        ttl_trade_time = compute_ttl_time_meet_requirement(trade_time_interval)
        ratio = ttl_good_time/ttl_trade_time
        # text to be exported to pdf
        msg += f"""
        Instrument: {instrument}
            - Total trade time: {ttl_trade_time}
            - Total time for which the (max) bid-ask spread and (min) size requirements are satisified (in seconds): {ttl_good_time}
            - Proportion: {round(ratio,2)}
            - Requirements Satisfied: {"Yes" if ratio >= required_time_ratio else "No"}
        
        """
    logging.info("DONE!\n")
    return msg

def mm_test(file_config):
    mm_config = pd.read_json(os.path.join(file_config['root'], file_config['json_config']))
    mm = pd.read_csv(os.path.join(file_config['root'], file_config['order']),
                     dtype={
                        "price": np.float64,
                        "size": np.float64,
                        "mid": np.float64
                    },
                    parse_dates = ['entered_ts','completed_ts']
    )
    res = mm_analyzer(mm, mm_config)
    return res

def write_mm_to_pdf(prob1_msg, opt_filename="mm_result.pdf"):
    pdf = FPDF()
    # Adding a page
    pdf.add_page(orientation="landscape")
    # set style and size of font 
    pdf.set_font("Helvetica", size = 15)
    pdf.write(txt = prob1_msg)
    # save the pdf
    pdf.output(opt_filename)


"""
Problem: markouts
Usually markouts compare the price at execution to the midpoint of the market at some specified future time after the trade. 
Here given `horizon_ticks`, from which we realize we will also have comparison to the midpoint of the market at time before trades.
"""
def write_markouts_to_pdf(markouts_data, text_pdf_title, plots_dir='./plots', front_msg='', opt_filename="markouts_result.pdf"):
    instruments = []
    pdf = FPDF()
    # Adding a page
    pdf.add_page(orientation="landscape")
    # set style and size of font 
    pdf.set_font("Helvetica", size = 15)
    if front_msg:
        pdf.write(txt=front_msg)
    for idx, (instrument, info) in enumerate(markouts_data.items()):
        instruments.append(instrument)
        order_cnt = info["analytics"]["tbl_order_cnt"]
        pdf.write(txt=f"""Results for {instrument}:\n""")
        # generate table for daily orders
        ## convert all data to text
        for col in order_cnt:
            order_cnt[col] = order_cnt[col].astype("str")
        tbl_data = order_cnt.to_records(index=False)
        line_height = pdf.font_size * 2.5
        col_width = pdf.epw / 4  # distribute content evenly
        header = order_cnt.columns

        for colname in header:
            pdf.multi_cell(col_width, line_height, colname, border=1, new_x="RIGHT", new_y="TOP", max_line_height=pdf.font_size)
        pdf.ln(line_height)

        for row in tbl_data:
            for datum in row:
                pdf.multi_cell(col_width, line_height, datum, border=1, new_x="RIGHT", new_y="TOP", max_line_height=pdf.font_size)
            pdf.ln(line_height)
        
        if idx != len(markouts_data)-1:
            pdf.add_page(orientation="landscape")
    
    # save pdf for text output
    pdf.output(text_pdf_title)

    # merge output
    pdfs = [text_pdf_title]
    for instrument in instruments:
        pngs = glob(f"{plots_dir}/{instrument}*.pdf")
        pngs.sort()
        pdfs+=pngs

    print("All pdfs to be merged: \n", pdfs, sep='')
    merge_pdfs(pdfs, opt_filename=opt_filename)

    # clear plots in plots dir
    for f in glob(f"{plots_dir}/*.pdf"):
        os.remove(f)

def merge_pdfs(pdfs:list, opt_filename):
    merger = PdfMerger()
    for pdf in pdfs:
        merger.append(pdf)

    merger.write(opt_filename)
    merger.close()

def run_mm(prob_file_config, opt_filename):
    print("Working on Problem MM.")
    prob1_msg = mm_test(prob_file_config)
    write_mm_to_pdf(prob1_msg, opt_filename=opt_filename)
    print(f"Problem MM Done and Output Saved at {opt_filename}. Logs available.")

def run_markouts(prob_file_config, plots_dir, msg, opt_filename, start_datetime=None, end_datetime=None):
    with open(os.path.join(prob_file_config["root"], prob_file_config["horizon_ticks"]), "r") as file:
        tmp = file.read()
    horizon_ticks = eval(tmp)

    markouts_data = read_n_process_markouts(prob_file_config)
    _ = analyze_orders(markouts_data, start_datetime=start_datetime, end_datetime=end_datetime, order_cnt_freq="1T",plot_freq="1T", plots_dir=plots_dir)
    _ = analyze_orders(markouts_data, plots_dir=plots_dir)
    _ = analyze_markouts(markouts_data, horizon_ticks = horizon_ticks, plots_dir=plots_dir)
    write_markouts_to_pdf(markouts_data, text_pdf_title="markouts_text.pdf", plots_dir=plots_dir, front_msg=msg, opt_filename=opt_filename)
    print(f"Problem Markouts Done and output Saved at {opt_filename}")

#prob2_file_config = './markouts/'
if __name__ == "__main__":
    prob1_file_config = {
        'root': './mm',
        'json_config': 'mm_config.json',
        'order': 'mm.csv'
    }
    prob2_file_config = {
        'root': './markouts/',
        'tickers': ["GMMAUSD", "LMDAUSD", "ZTAUSD", "BTAUSD"],
        'horizon_ticks': 'horizon_ticks'
    }
    # solve problem mm
    prob1_opt_filename = "mm_result.pdf"
    run_mm(prob1_file_config, opt_filename=prob1_opt_filename)
   
    # solve problem markouts
    print("Working on Problem Markouts.")
    prob2_opt_filename="markouts_result.pdf"
    markout_msg = """
    Problem -  Markouts

    General Findings:
        - bid-ask spreads exhibit a U-shaped pattern, with spreads wider at the start and end of the trading day, whilst spreads are tighter in the middle of the day.

    """
    ## read period in user's interest (other code re-factoring options: command-line options using argparse)
    period_to_query = json.load(open("markouts_periods_to_query.json"))
    start_datetime = period_to_query.get("start_datetime",None)
    end_datetime = period_to_query.get("end_datetime",None)
    plots_dir = "./plots"
    run_markouts(prob2_file_config, plots_dir, markout_msg, prob2_opt_filename, start_datetime=start_datetime, end_datetime=end_datetime)
    
    # merge answers to both questions
    merge_pdfs([prob1_opt_filename, prob2_opt_filename], opt_filename="Merged Deliverable.pdf")
    print("Merged result exported!")