"""
Example usage of the DocumentParser to extract atomic facts from news paragraphs.

This script demonstrates:
1. Loading the LLM model
2. Creating a sample Excel file with test data
3. Using DocumentParser to extract atomic facts
4. Examining the results
"""

import asyncio
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Import the LLM model configuration
from models.models import get_default_model

# Import the document parser
from demo.doc_parser import DocumentParser


async def create_sample_excel(sample_path: str = "sample_news.xlsx"):
    """
    Create a sample Excel file with test data for demonstration.
    """
    sample_data = {
        'date': [
            datetime(2020, 1, 9).strftime('%Y-%m-%d'),
            datetime(2020, 1, 23).strftime('%Y-%m-%d'),
            datetime(2020, 1, 27).strftime('%Y-%m-%d'),
            datetime(2020, 1, 28).strftime('%Y-%m-%d'),
            datetime(2020, 1, 30).strftime('%Y-%m-%d'),
            datetime(2020, 2, 6).strftime('%Y-%m-%d'),
            datetime(2020, 2, 7).strftime('%Y-%m-%d'),
            datetime(2020, 2, 9).strftime('%Y-%m-%d'),
            datetime(2020, 2, 10).strftime('%Y-%m-%d'),
            datetime(2020, 2, 11).strftime('%Y-%m-%d'),
        ],
        'lead_paragraph': [
            'HONG KONG — Chinese researchers say they have identified a new virus behind an illness that has infected dozens of people across Asia, setting off fears in a region that was struck by a deadly epidemic 17 years ago.',
            'The spread of a mysterious respiratory virus has prompted the authorities to limit travel in cities in China, including Wuhan, where the disease was first found last month. It has since spread across the nation and to at least 10 other countries.',
            'U.S. stock futures are down sharply this morning on fears about the coronavirus outbreak. More below. (Want this in your inbox each morning? Sign up here.)',
            'As President Trump’s lawyers open the third day of their defense today, an important question hangs over Washington: Will the Republican-controlled Senate agree to hear testimony from witnesses?.When I flew to China at the beginning of January to teach a three-week college class, the Wuhan coronavirus was barely on anyone’s radar. By the time I got back to New York last Friday, it was front page news around the world, with more than 800 cases in China and 26 deaths; by Monday, the total was at least 80 deaths.. .Hello. On today’s agenda: Apple is scheduled to report earnings today. (Want this in your inbox each morning? Sign up here.)',
            'The World Health Organization declared a global health emergency on Thursday as the coronavirus outbreak spread well beyond China, where it emerged last month.',
            'ANGERS, France — The relentless whir of machines echoing across a cavernous French factory floor this week is an unexpected result of the deadly virus that has nearly paralyzed cities in China and other parts of Asia. The company, Kolmi Hopen, happens to make an item that is suddenly one of the world’s hottest commodities: the medical face mask.',
            'This year’s edition of Art Basel Hong Kong, one of the most important destinations in the international art market calendar, has been canceled, with organizers citing the ‘‘sudden and widespread outbreak’’ of the coronavirus in China. The fair, held at the Hong Kong Convention and Exhibition Center, and featuring premier galleries from Asia and beyond, was to run March 17  through March 21....Japan already had several confirmed coronavirus cases when a giant cruise ship arrived at the port of Yokohama last week..An alliance between Saudi Arabia and Russia has helped prop up oil prices for the last three years. But the two big oil producers were not in perfect harmony this week, as they have tried to recalibrate production targets to cope with reduced demand from China, whose economy has been crippled by the coronavirus epidemic.',
            'BEIJING — The coronavirus epidemic in China surpassed a grim milestone on Sunday with a death toll that exceeds that of the SARS outbreak 17 years ago, a development that coincided with news that World Health Organization experts might soon be in the country to help stanch the crisis.',
            'As many people across China return to work today after an already-extended Lunar New Year break, the country is confronting two bleak statistics:',
            'NUSA DUA, Indonesia — The family from Shanghai was vacationing in Singapore last month when they learned that the new coronavirus had arrived there from China..When it comes to making decisions that involve risks, we humans can be irrational in quite systematic ways — a fact that the psychologists Amos Tversky and Daniel Kahneman famously demonstrated with the help of a hypothetical situation, eerily apropos of today’s coronavirus epidemic, that has come to be known as the Asian disease problem..A woman sick from the coronavirus was released from a San Diego hospital this week after a labeling error on samples to be tested for the virus led officials to incorrectly indicate that she was not infected, federal authorities said on Tuesday..Chinese health officials said today that the death toll from the coronavirus had passed 1,000. Here are the latest updates and maps of where the virus has reached.',
        ]
    }
    
    df_sample = pd.DataFrame(sample_data)
    df_sample.to_excel(sample_path, index=False)
    print(f"✅ Sample Excel file created: {sample_path}")
    return sample_path


async def main():
    """Main demo function."""
    
    print("=" * 70)
    print("🔬 ATOMIC FACTS EXTRACTION DEMO")
    print("=" * 70)
    
    # Step 1: Get the default LLM model
    print("\n📦 Step 1: Loading LLM model...")
    llm_model = get_default_model()
    print("✅ LLM model loaded successfully")
    
    # Step 2: Create sample Excel file
    print("\n📝 Step 2: Creating sample Excel file...")
    sample_excel_path = "sample_news.xlsx"
    await create_sample_excel(sample_excel_path)
    
    # Step 3: Initialize DocumentParser
    print("\n🔧 Step 3: Initializing DocumentParser...")
    parser = DocumentParser(llm_model=llm_model)
    print("✅ DocumentParser initialized")
    
    # Step 4: Parse the Excel file
    print("\n⚙️  Step 4: Extracting atomic facts from paragraphs...")
    print("   (Using parallel batch processing with batch_size=3)")
    print("   (Applying post-processing: date normalization, deduplication, relevance filtering)")
    print("-" * 70)
    
    output_excel_path = "sample_news_with_factoids.xlsx"
    result_df = await parser.parse_excel(
        input_excel_path=sample_excel_path,
        output_excel_path=output_excel_path,
        batch_size=3,  # Process 3 paragraphs in parallel per batch
        apply_post_processing=True  # Enable post-processing
    )
    
    # Step 5: Display the results
    print("\n" + "=" * 70)
    print("📊 EXTRACTION RESULTS")
    print("=" * 70)
    
    for idx, row in result_df.iterrows():
        print(f"\n📅 Date: {row['date']}")
        print(f"📄 Paragraph: {row['lead_paragraph'][:100]}...")
        print(f"✨ Extracted Atomic Facts ({len(row['factoids_g_truth'])} facts):")
        
        if isinstance(row['factoids_g_truth'], list) and len(row['factoids_g_truth']) > 0:
            for i, fact in enumerate(row['factoids_g_truth'], 1):
                print(f"   {i}. {fact}")
        else:
            print("   [No facts extracted]")
    
    print("\n" + "=" * 70)
    print(f"✅ Results saved to: {output_excel_path}")
    print("=" * 70)
    
    # Display summary statistics
    print("\n📈 SUMMARY STATISTICS")
    print("-" * 70)
    total_factoids = sum(len(facts) if isinstance(facts, list) else 0 
                        for facts in result_df['factoids_g_truth'])
    print(f"Total rows processed: {len(result_df)}")
    print(f"Total atomic facts extracted: {total_factoids}")
    print(f"Average factoids per paragraph: {total_factoids / len(result_df):.1f}")


if __name__ == "__main__":
    asyncio.run(main())
