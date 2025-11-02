from example import train_gdp, test_gdp

if __name__ == "__main__":
    # Train
    model, dataset = train_gdp(num_workers=4)
    
    # Test
    test_gdp(model, dataset)
