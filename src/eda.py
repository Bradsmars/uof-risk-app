from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  #
import matplotlib.pyplot as plt


def map_highrisk_to_binary(target_series):
    """
    here im convert the highrisk target column into 0/1 binary values
    Mapping:
      - minor -> 0
      - highRisk -> 1
      - if already numeric, coerce to 0/1
    """
    if target_series.dtype == object:
        mapping = {"Minor": 0, "HighRisk": 1, "0": 0, "1": 1}
        out = target_series.map(mapping)
        # fallback if something unmapped appears
        out = pd.to_numeric(out, errors="coerce").fillna(0)
        return out.astype(int)
    return pd.to_numeric(target_series, errors="coerce").fillna(0).astype(int)


def save_bar(series_or_frame, title, out_path):
    """ saving  bar charts"""
    plt.figure()
    series_or_frame.plot(kind="bar")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_eda_simpler(dataframe, output_directory = "eda_simple_plots"):
    """
    saves pngs to output_directory and returns the absolute folder path.
    """
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)


    # code adapted from: https://www.machinelearningplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python/#32.-Pie-Chart
    # inspired by: research methods 
    target_binary = map_highrisk_to_binary(dataframe["highrisk"])

    # 1) Target class counts (pie chart) with counts and percentages 
    minor_vs_highrisk_counts = target_binary.value_counts().reindex([0, 1], fill_value=0) # ensure both classes are present
    class_labels_descriptive = ["Minor (0)", "HighRisk (1)"] # more descriptive labels
    
    # Draw Plot
    fig, ax = plt.subplots(figsize=(12, 7), subplot_kw=dict(aspect="equal"), dpi=80)
    incident_data = minor_vs_highrisk_counts.values
    class_categories = class_labels_descriptive
    explode_sections = [0, 0.1]  # Slightly separate the HighRisk slice for emphasis
    
    def format_percentage_and_count(percentage, all_values):
        absolute_count = round(percentage/100.*np.sum(all_values))
        return "{:.1f}% ({:d})".format(percentage, absolute_count)
    
    wedge_slices, text_labels, percentage_texts = ax.pie(incident_data, 
                                      autopct=lambda pct: format_percentage_and_count(pct, incident_data),
                                      textprops=dict(color="black"), 
                                      colors=["#2E8B57", "#DC143C"],  # Sea green for Minor, Crimson for HighRisk
                                     startangle=140,
                                     explode=explode_sections)
    
    # Decoration
    ax.legend(wedge_slices, class_categories, title="Incident Risk Level", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.setp(percentage_texts, size=10, weight=700)
    ax.set_title("Target Class Distribution: Pie Chart")
    plt.tight_layout()
    plt.savefig(output_path / "01_target_counts_pie.png", dpi=150)
    plt.close()

    # 2) HighRisk rate by year (line)
    
    # if year exists in dataframe columns
    if "year" in dataframe.columns:
        # then create dataframe with year and target
        temporary_dataframe = pd.DataFrame({"year": dataframe["year"].astype(str), "y": target_binary})
        # Calculate mean rate by year
        rate_by_year = temporary_dataframe.groupby("year")["y"].mean().sort_index()
        
        # Plot the results using pandas plot() directly
        ax = rate_by_year.plot(ylabel='High-Risk Rate', 
                              figsize=(10, 6), 
                              marker='o',
                              title="Share of HighRisk by year",
                              xlabel="Year")
        
        # formatting y-axis as percentages
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(output_path / "02HighRisk_rate_by_year.png", dpi=150)
        plt.close()




    # highrisk by subject_vulnerability
    if "subject_vulnerable" in dataframe.columns:
        vulnerability = pd.to_numeric(dataframe["subject_vulnerable"], errors="coerce")
        # make a small categorical with 0/1/“missing” for plotting
        vulnerable_cat = vulnerability.map({0: "0", 1: "1"}).astype("string")
        vulnerable_cat = vulnerable_cat.fillna("missing")

        order = ["0", "1", "missing"]  # consistent bar order
        mask = vulnerable_cat.isin(order)
        rate_by_vuln = (
            pd.Series(target_binary[mask].values, index=vulnerable_cat[mask])
            .groupby(level=0).mean()
            .reindex(order)
        )
        save_bar(
            rate_by_vuln,
            "average HighRisk rate by subject vulnerability",
            output_path / "04_rate_by_subject_vulnerability.png"
        )
        
    # average HighRisk rate by police_force
    if "police_force" in dataframe.columns:
        force = dataframe["police_force"].astype(str)
        force = force.fillna("missing")
        vol = force.value_counts()
        top_forces = vol.head(10).index
        mask = force.isin(top_forces)
        rate_by_force = (
            pd.Series(target_binary[mask].values, index=force[mask])
            .groupby(level=0)
            .mean()
            .reindex(top_forces)
        )
        save_bar(rate_by_force, "mean HighRisk rate by police_force (top 10 police forces)",
                 output_path / "05_rate_by_force_top10.png")


    # 5) HighRisk count by ethnicity (top 10 categories)
    if "person_perceived_ethnicity" in dataframe.columns:
        ethnicity = dataframe["person_perceived_ethnicity"].astype(str).str.strip()
        ethnicity = ethnicity.fillna("missing").replace("None", "missing values") 
        top_ethnicities = ethnicity.value_counts().head(10).index
        mask = ethnicity.isin(top_ethnicities)
        
        # calculate counts instead of rates
        count_by_ethnicity = (
            pd.Series(target_binary[mask].values, index=ethnicity[mask])
            .groupby(level=0)
            .sum()  # Changed from .mean() to .sum() for counts
            .reindex(top_ethnicities)
        )
        
        # Creating custom plot with better visibility
        plt.figure(figsize=(12, 8))
        bars = count_by_ethnicity.plot(kind="bar", color='crimson', alpha=0.8)
        
        # Add value labels on top of bars (now showing counts)
        for i, v in enumerate(count_by_ethnicity.values):
            plt.text(i, v + v*0.01, f'{int(v)}', ha='center', va='bottom', fontweight='bold')
        
        plt.title("HighRisk count by ethnicity", fontsize=14, fontweight='bold')  # Updated title
        plt.ylabel("High-Risk Count")  # Updated y-label
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, max(count_by_ethnicity.values) * 1.15)  # Add space for labels
        
        # adding grid for easier comparison
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "06_count_by_ethnicity.png", dpi=150)  # Updated filename
        plt.close()
    
    
    
    # Highrisk rate by age
    if "person_perceived_age" in dataframe.columns:
        age = dataframe["person_perceived_age"].astype(str).str.strip()
        age = age.fillna("missing")
        top_ages = age.value_counts().head(10).index
        mask = age.isin(top_ages)
        rate_by_age = (
            pd.Series(target_binary[mask].values, index=age[mask])
            .groupby(level=0)
            .mean()
            .reindex(top_ages)
        )
        save_bar(rate_by_age, "HighRisk rate by age (top 10)",
                 output_path / "07_rate_by_age_top5.png")

    #  4) HighRisk rate by police_force (top 10 by volume)
    if "police_force" in dataframe.columns:
        force = dataframe["police_force"].astype(str)
        force = force.fillna("missing")
        vol = force.value_counts()
        top_forces = vol.head(10).index
        mask = force.isin(top_forces)
        rate_by_force = (
            pd.Series(target_binary[mask].values, index=force[mask])
            .groupby(level=0)
            .mean()
            .reindex(top_forces)
        )
        save_bar(rate_by_force, "HighRisk rate by police_force (top 10 by volume)",
                 output_path / "08_rate_by_force_top10.png")
    

    # if police_force exists in dataframe columns
    if "police_force" in dataframe.columns:
        # Getting the top 5 police forces
        top_forces_counts = dataframe["police_force"].astype(str).value_counts().head(5)
        # Top 5 police forces by incident count
        save_bar(top_forces_counts, "top 5 police forces by incident count",
                 output_path / "09_forces_counts.png")
        


    # HighRisk rate by CED highest use (top 8)
    if "ced_highest_use" in dataframe.columns:
        ced = dataframe["ced_highest_use"].astype(str).str.strip()
        # Exclude "missing" entries
        ced = ced.where(ced.str.lower() != "missing")
        # Get top 8 CEDs by volume
        top_ced = ced.value_counts().head(8).index
        mask = ced.isin(top_ced)
        rate_by_ced = (
            pd.Series(target_binary[mask].values, index=ced[mask])
            .groupby(level=0)
            .mean()
            .reindex(top_ced)
        )
        save_bar(rate_by_ced, "HighRisk rate by CED highest use (top 8)",
                output_path / "10_rate_by_ced_top8.png")
        
    
    # HighRisk rate by year, split by any_firearm (yes/no)
    if "year" in dataframe.columns and "any_firearm" in dataframe.columns:
        year = dataframe["year"].astype(str)
        any_firearm = dataframe["any_firearm"].astype(int)
        mask = any_firearm.isin([1, 0])
        temp_dataframe = pd.DataFrame({"year": year[mask], "any firearm": any_firearm[mask], "highrisk": target_binary[mask]})
        pivot = temp_dataframe.pivot_table(values="highrisk", index="year", columns="any firearm", aggfunc="mean").sort_index()
        plt.figure()
        pivot.plot(marker="o")
        plt.title("HighRisk rate by year (split by any_firearm)")
        plt.ylabel("Rate (HighRisk)")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(output_path / "11_rate_by_year_split_any_firearm.png", dpi=150)
        plt.close()
    
    
    
        # 14) HighRisk rate by gender
    # 14) HighRisk count by gender
    if "person_perceived_gender" in dataframe.columns:
        gender = dataframe["person_perceived_gender"].astype(str).str.strip()
        ethnicity = ethnicity.fillna("missing").replace("None", "missing values")
        top_genders = gender.value_counts().head(10).index
        mask = gender.isin(top_genders)
        
        # Calculating counts instead of rates
        count_by_gender = (
            pd.Series(target_binary[mask].values, index=gender[mask])
            .groupby(level=0)
            .sum()  # sum for counts
            .reindex(top_genders)
        )
        
        # creating a custom plot with better visibility
        plt.figure(figsize=(12, 8))
        bars = count_by_gender.plot(kind="bar", color='crimson', alpha=0.8)
        
        # adding value labels on top of bars (now showing counts)
        for i, v in enumerate(count_by_gender.values):
            plt.text(i, v + v*0.01, f'{int(v)}', ha='center', va='bottom', fontweight='bold')
        
        plt.title("HighRisk count by gender", fontsize=14, fontweight='bold')  # Updated title
        plt.ylabel("High-Risk Count")  # Updated y-label
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, max(count_by_gender.values) * 1.15)  # Add space for labels
        
        # Remove percentage formatting (now showing raw counts)
        # plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # add grid for easier comparison
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "014_count_by_gender.png", dpi=150)  # Updated filename
        plt.close()
        
           
    if "person_perceived_ethnicity" in dataframe.columns:
   
        top_ethnicities_counts = dataframe["person_perceived_ethnicity"].astype(str).value_counts().head()
        top_ethnicities_counts = top_ethnicities_counts.fillna("missing")
  
        save_bar(top_ethnicities_counts, "top ethnicities by incident count",
                 output_path / "12_ethnicities_counts.png")
    
          
    if "person_perceived_gender" in dataframe.columns:
       
        top_genders_counts = dataframe["person_perceived_gender"].astype(str).value_counts().head()
        top_genders_counts = top_genders_counts.fillna("missing")

        save_bar(top_genders_counts, "top genders by incident count",
                 output_path / "13_genders_counts.png")

    return str(output_path.resolve())
