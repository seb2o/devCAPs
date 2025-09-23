def pretty_print_dataset_dict(dict):
    for subject_path, sessions_dict in dict.items():
        print(f"Subject: {subject_path.name}")
        for session_path, session_data in sessions_dict.items():
            print(f"  Session: {session_path.name}")
            bolds = [b.name for b in session_data['bolds']]
            transform = session_data['transform'].name
            print(f"    Transform: {transform}")
            for runid, run in enumerate(bolds):
                print(f"    Run {runid}: {run}")