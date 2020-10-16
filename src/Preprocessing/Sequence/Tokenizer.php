<?php
namespace Rindow\NeuralNetworks\Preprocessing\Sequence;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;
use InvalidArgumentException;

class Preprocessor
{
    use GenericUtils;
    protected $mo;

    public function __construct($mo, array $options=null)
    {
        $this->mo = $mo;
        extract($this->extractArgs([
            'num_words'=>null,
            'filters'=>'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            'lower'=>true,
            'split'=>" ",
            'char_level'=>false,
            'oov_token'=>null,
            'document_count'=>0,
        ],$options));
    }

    public function fitOnTexts($texts) : void
    {
    }
    //public function fitOnSequences($sequences) : void
    //{
    //}
    public function textsToSequences($texts)
    {
    }
    //public function sequencesToTexts($sequences)
    //{
    //}
    public function wordToIndex(string $word) : int
    {
    }
    public function indexToWord(int $index) : string
    {
    }
    public function vocabularySize() : int
    {
    }
}
