# UIS_Trade-off

This repository is the official implementation of the paper:
* On the Utility-Informativeness-Security Trade-off in Discrete Task-Oriented Semantic Communication
* Authors: Anbang Zhang, Yanhu Wang, and Shuaishuai Guo

## Design Philosophy

### This Work: Synergistic Alignment of Learning and Communication Objectives
![9bb7b791ab362947901ad0c3439e530](https://github.com/zab0613/UIS_Trade-off/assets/117052094/e6d80e85-9532-4b05-8777-3b59f61c394c)

In detail, most existing frameworks for task-oriented semantic communications predominantly concentrate on individual aspects such as edge inference performance, transmission enhancement, or security improvement. Rarely do these frameworks address all three elements concurrently. Our UIS-ToSC framework stands out by combining edge inference utility, informativeness, and enhanced security. By incorporating the information bottleneck principle, vector quantization loss, MSE loss, and perceptual loss, each focusing on different metrics, one can find a more nuanced balance between these competing objectives and tailor the training process to domain-specific needs more effectively. Moreover, we introduce adversarial learning into UIS-ToSC ensuring that the designed codebook possesses intrinsic security properties, setting our work apart from conventional methods.

## Usage

```python
import foobar

# returns 'words'
foobar.pluralize('word')

# returns 'geese'
foobar.pluralize('goose')

# returns 'phenomenon'
foobar.singularize('phenomena')
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
