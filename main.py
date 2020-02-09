import argparse
from baseline_trainer import BaselineNetTrainer
from coattention_trainer import CoattentionNetTrainer


if __name__ == "__main__":
    # Feel free to add more args, or change/remove these.
    parser = argparse.ArgumentParser(description='Load VQA.')
    parser.add_argument('--model', type=str, choices=['simple', 'coattention'], default='simple')
    parser.add_argument('--train_image_dir', type=str, default='./data/train2014')
    parser.add_argument('--train_question_path', type=str, default='./data/v2_OpenEnded_mscoco_train2014_questions.json')
    parser.add_argument('--train_annotation_path', type=str, default='./data/v2_mscoco_train2014_annotations.json')
    parser.add_argument('--test_image_dir', type=str, default='./data/val2014')
    parser.add_argument('--test_question_path', type=str, default='./data/v2_OpenEnded_mscoco_val2014_questions.json')
    parser.add_argument('--test_annotation_path', type=str, default='./data/v2_mscoco_val2014_annotations.json')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--num_data_loader_workers', type=int, default=16)
    args = parser.parse_args()

    if args.model == "simple":
        trainer_class = BaselineNetTrainer
    elif args.model == "coattention":
        trainer_class = CoattentionNetTrainer
    else:
        raise ModuleNotFoundError()

    trainer = trainer_class(train_image_dir=args.train_image_dir,
                                                train_question_path=args.train_question_path,
                                                train_annotation_path=args.train_annotation_path,
                                                test_image_dir=args.test_image_dir,
                                                test_question_path=args.test_question_path,
                                                test_annotation_path=args.test_annotation_path,
                                                batch_size=args.batch_size,
                                                num_epochs=args.num_epochs,
                                                num_data_loader_workers=args.num_data_loader_workers)
    trainer.train()
