<?php
namespace Rindow\NeuralNetworks\Data\Sequence;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;
use InvalidArgumentException;
use ArrayObject;

class Tokenizer
{
    use GenericUtils;
    protected $mo;
    protected $analyzer;
    protected $numWords;
    protected $filters;
    protected $lower;
    protected $split;
    protected $charLevel;
    protected $oovToken;
    protected $documentCount;
    protected $wordCounts = [];
    protected $wordToIndex = [];
    protected $indexToWord = [];
    protected $filtersMap;
    protected $specialsMap;

    public function __construct($mo, array $options=null)
    {
        $this->mo = $mo;
        extract($this->extractArgs([
            'num_words'=>null,
            'filters'=>"!\"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n",
            'specials'=>null,
            'lower'=>true,
            'split'=>" ",
            'char_level'=>false,
            'oov_token'=>null,
            'document_count'=>0,
        ],$options));
        $this->numWords = $num_words;
        $this->filters = $filters;
        $this->specials = $specials;
        $this->lower = $lower;
        $this->split = $split;
        $this->charLevel = $char_level;
        $this->oovToken = $oov_token;
        $this->documentCount = $document_count;
    }

    protected function textToWordSequence(
        $text,$filters,$specials,$lower,$split)
    {
        if($lower)
            $text = mb_strtolower($text);
        $text = mb_str_split($text);
        if($this->filtersMap) {
            $map = $this->filtersMap;
        } else {
            if($filters) {
                $map = $this->filtersMap = array_flip(str_split($filters));
            } else {
                $map = [];
            }
        }
        if($this->specialsMap) {
            $specialsMap = $this->specialsMap;
        } else {
            if($specials) {
                $specialsMap = $this->specialsMap = array_flip(str_split($specials));
            } else {
                $specialsMap = [];
            }
        }
        $func = function($c) use ($map,$specialsMap,$split) {
            if($specialsMap && array_key_exists($c,$specialsMap)) {
                $c = $split.$c.$split;
            } else {
                $c = array_key_exists($c,$map) ? $split : $c;
            }
            return $c;
        };
        $text = implode('',array_map($func,$text));
        $seq = array_values(array_filter(explode(' ',$text)));
        return $seq;
    }

    public function fitOnTexts(iterable $texts) : void
    {
        foreach ($texts as $text) {
            $this->documentCount++;
            if($this->analyzer) {
                $seq = $this->analyzer($text);
            } else {
                $seq = $this->textToWordSequence(
                    $text,$this->filters,$this->specials,$this->lower,$this->split);
            }
            foreach ($seq as $w) {
                if(array_key_exists($w,$this->wordCounts)) {
                    $this->wordCounts[$w]++;
                } else {
                    $this->wordCounts[$w] = 1;
                }
            }
        }
        arsort($this->wordCounts);
        if($this->oovToken) {
            $this->wordCounts = array_merge([$this->oovToken=>1],$this->wordCounts);
        }
        $idx = 1;
        foreach ($this->wordCounts as $key => $value) {
            $this->wordToIndex[$key] = $idx;
            $this->indexToWord[$idx] = $key;
            $idx++;
        }
    }
    //public function fitOnSequences($sequences) : void
    //{
    //}
    public function textsToSequences(iterable $texts) : iterable
    {
        $sequences = new ArrayObject();
        foreach($this->textsToSequencesGenerator($texts) as $seq) {
            $sequences->append($seq);
        }
        return $sequences;
    }

    public function textsToSequencesGenerator(iterable $texts) : iterable
    {
        $numWords = $this->numWords;
        $oovTokenIndex = ($this->oovToken) ?
            $this->wordToindex[$this->oovToken] : null;
        foreach ($texts as $text) {
            if($this->analyzer) {
                $seq = $this->analyzer($text);
            } else {
                $seq = $this->textToWordSequence(
                    $text,$this->filters,$this->specials,$this->lower,$this->split);
            }
            $vect = [];
            foreach ($seq as $w) {
                if(array_key_exists($w,$this->wordToIndex)) {
                    $i = $this->wordToIndex[$w];
                    if($numWords==null || $i<$numWords) {
                        $vect[] = $i;
                    } else {
                        if($oovTokenIndex) {
                            $vect[] = $oovTokenIndex;
                        }
                    }
                } else {
                    if($oovTokenIndex) {
                        $vect[] = $oovTokenIndex;
                    }
                }
            }
            yield $vect;
        }
    }

    public function sequencesToTexts(iterable $sequences) : iterable
    {
        $texts = new ArrayObject();
        foreach($this->sequencesToTextsGenerator($sequences) as $text) {
            $texts->append($text);
        }
        return $texts;
    }

    public function sequencesToTextsGenerator(iterable $sequences) : iterable
    {
        if(!is_iterable($sequences)) {
            throw new InvalidArgumentException('sequences must be list of sequence.');
        }
        $numWords = $this->numWords;
        $oovTokenIndex = ($this->oovToken) ?
            $this->wordToindex[$this->oovToken] : null;
        foreach ($sequences as $seq) {
            if(!is_iterable($seq)) {
                throw new InvalidArgumentException('sequences must be list of sequence.');
            }
            $vect = [];
            foreach ($seq as $num) {
                if(!is_int($num)) {
                    throw new InvalidArgumentException('sequence includes "'.gettype($num).'". it must be numeric.');
                }
                if(array_key_exists($num,$this->indexToWord)) {
                    $w = $this->indexToWord[$num];
                    if($numWords==null || $num<$numWords) {
                        $vect[] = $w;
                    } else {
                        if($oovTokenIndex) {
                            $vect[] = $this->oovToken;
                        }
                    }
                } else {
                    if($oovTokenIndex) {
                        $vect[] = $this->oovToken;
                    }
                }
            }
            $vect = implode($this->split,$vect);
            yield $vect;
        }
    }

    public function documentCount() : int
    {
        return $this->documentCount;
    }

    public function wordCounts() : array
    {
        return $this->wordCounts;
    }

    public function getWords()
    {
        return $this->indexToWord;
    }

    public function numWords(bool $internal=null) : int
    {
        if($internal||$this->numWords===null) {
            return count($this->indexToWord)+1;
        } else {
            return min(count($this->indexToWord)+1,$this->numWords);
        }
    }

    public function wordToIndex(string $word) : int
    {
        if(!array_key_exists($word,$this->wordToIndex))
            return null;
        return $this->wordToIndex[$word];
    }

    public function indexToWord(int $index) : string
    {
        if(!array_key_exists($index,$this->indexToWord))
            return null;
        return $this->indexToWord[$index];
    }
}