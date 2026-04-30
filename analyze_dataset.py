"""
Diagnostic script to analyze the COVID dataset before processing.
This helps you understand how much work your script will do.
"""
import pandas as pd
import ast

print("=" * 80)
print("DATASET ANALYSIS")
print("=" * 80)

try:
    # Load the dataset
    print("\nLoading dataset...")
    news_covid = pd.read_pickle("./itext2kg-1.0.0/datasets/atom/nyt_news/2020_nyt_COVID_last_version_ready.pkl")
    
    print(f"✓ Dataset loaded successfully")
    print(f"\nDataset shape: {news_covid.shape}")
    print(f"Columns: {list(news_covid.columns)}")
    
    # Convert factoids if needed
    if isinstance(news_covid['factoids_g_truth'][0], str):
        news_covid["factoids_g_truth"] = news_covid["factoids_g_truth"].apply(lambda x: ast.literal_eval(x))
    
    # Analyze the data
    total_rows = len(news_covid)
    total_dates = news_covid['date'].nunique()
    
    print(f"\n" + "=" * 80)
    print("WORKLOAD ESTIMATION")
    print("=" * 80)
    print(f"Total rows: {total_rows}")
    print(f"Unique dates: {total_dates}")
    
    # Group by date and count facts
    grouped = news_covid.groupby("date")["factoids_g_truth"].sum()
    facts_per_date = grouped.apply(len)
    
    total_facts = facts_per_date.sum()
    avg_facts = facts_per_date.mean()
    max_facts = facts_per_date.max()
    min_facts = facts_per_date.min()
    
    print(f"\nTotal atomic facts across all dates: {total_facts}")
    print(f"Average facts per date: {avg_facts:.1f}")
    print(f"Min facts per date: {min_facts}")
    print(f"Max facts per date: {max_facts}")
    
    print(f"\n" + "=" * 80)
    print("PROCESSING TIME ESTIMATES")
    print("=" * 80)
    
    # Show what happens with different configurations
    configs = [
        (2, 3, "MINIMAL TEST (recommended for laptop)"),
        (5, 10, "SMALL TEST"),
        (10, 20, "MEDIUM TEST"),
        (20, None, "ORIGINAL (FULL) - DAYS OF PROCESSING!"),
    ]
    
    for max_dates, max_facts_per_date, label in configs:
        dates_to_process = min(max_dates, total_dates)
        
        if max_facts_per_date is None:
            # Calculate actual facts for first max_dates
            # actual_facts = facts_per_date.head(dates_to_process).sum()
            actual_facts = dates_to_process * total_facts
        else:
            actual_facts = dates_to_process * max_facts_per_date
        
        # Estimate processing time (very rough)
        # Assume ~10 seconds per fact with local Ollama on average hardware
        estimated_seconds = actual_facts * 10
        estimated_minutes = estimated_seconds / 60
        estimated_hours = estimated_minutes / 60
        
        print(f"\n{label}:")
        print(f"  Dates: {dates_to_process}")
        print(f"  Facts per date: {max_facts_per_date if max_facts_per_date else 'ALL'}")
        print(f"  Total facts to process: {actual_facts}")
        print(f"  Estimated time: ", end="")
        if estimated_hours >= 1:
            print(f"{estimated_hours:.1f} hours")
        else:
            print(f"{estimated_minutes:.1f} minutes")
    
    print(f"\n" + "=" * 80)
    print("SAMPLE DATA (First 3 dates)")
    print("=" * 80)
    
    for i, (date, facts) in enumerate(grouped.head(3).items()):
        print(f"\nDate {i+1}: {date}")
        print(f"  Number of facts: {len(facts)}")
        print(f"  Sample facts:")
        for j, fact in enumerate(facts[:3]):
            print(f"    {j+1}. {fact}")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print("\n⚠️  WARNING: Processing the full dataset will take DAYS!")
    print("\n✓ RECOMMENDED: Start with MAX_DATES=2, MAX_FACTS_PER_DATE=3")
    print("  This will process only 6 facts and take ~1-2 minutes")
    print("\n✓ Then gradually increase to test performance on your hardware")
    print("=" * 80)
    
except FileNotFoundError:
    print("\n❌ ERROR: Dataset file not found!")
    print("Expected location: ./itext2kg-1.0.0/datasets/atom/nyt_news/2020_nyt_COVID_last_version_ready.pkl")
    print("\nMake sure you're running this from the project root directory.")
except Exception as e:
    print(f"\n❌ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()