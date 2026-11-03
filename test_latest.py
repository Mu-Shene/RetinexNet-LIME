from main import build_parser, run_test


def main(argv=None):
    parser = build_parser(default_phase='test')
    parser.description = __doc__
    args = parser.parse_args(argv)
    if args.phase != 'test':
        parser.error('test_latest.py only supports --phase=test; use main.py to train')
    return run_test(args)


if __name__ == '__main__':
    main()
