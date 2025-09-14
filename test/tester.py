from test.test_utils import test_loss,evaluate_model,visualize_predictions

def test_model(model, test_loader, criterion, device):
    loss = test_loss(model, test_loader, criterion, device)
    print("Loss:",loss)
    visualize_predictions(model, test_loader, num_samples=3)
    avg_pixel_error, avg_rand_error= evaluate_model(model, test_loader)
    print("avg_pixel_error:",avg_pixel_error)
    print("avg_rand_error",avg_rand_error)